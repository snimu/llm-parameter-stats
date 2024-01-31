"""Sparsify models by removing parameters within a band of values."""
from __future__ import annotations

import sys
import os 
import urllib
import shutil
import itertools
from collections.abc import Sequence

import zipfile
import numpy as np
import torch 
from torch import nn
from tqdm import tqdm
import rich
try:
    from beartype import beartype
except ImportError:
    pass
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from packaging import version
import pandas as pd


STEPS = torch.tensor([0] + [2**i for i in range(10)] + [i * 1000 for i in range(1, 144)])
PYTHIA_BATCH_SIZE = 2e6  # 2 million
MODEL_SIZES = [
        "70m", "70m-deduped",
        "160m", "160m-deduped",
        "410m", "410m-deduped", 
        "1b", "1b-deduped",
        "1.4b", "1.4b-deduped",
        "2.8b", "2.8b-deduped",
        # "6.9b", "6.9b-deduped",
        # "12b", "12b-deduped",
]


def save_beartype(func):
    if version.parse(sys.version.split(" ")[0]) >= version.parse("3.10"):
        return beartype(func)
    else:
        return func


def pairwise(x):
    try:
        return itertools.pairwise(x)
    except AttributeError:
        return zip(x[:-1], x[1:])


def save_inference_mode(func):
    if hasattr(torch, "inference_mode") and callable(torch.inference_mode):
        return torch.inference_mode()(func)
    else:
        return torch.no_grad()(func)


@save_beartype
def get_chinchilla_optimal_steps(
        model_size: str,
) -> int:
    if "m" in model_size: 
        model_size = float(model_size.split("m")[0]) * 1e6
    elif "b" in model_size:
        model_size = float(model_size.split("b")[0]) * 1e9
    else:
        raise ValueError(f"Model size {model_size} not supported.")

    num_samples_chinchilla_optimal = 20 * model_size
    num_steps_chinchilla_optimal = int(num_samples_chinchilla_optimal / PYTHIA_BATCH_SIZE)
    return num_steps_chinchilla_optimal


@save_beartype
def get_percentage_of_chinchilla_optimal(
        model_size: str,
        steps: Sequence[int] = STEPS,
) -> torch.Tensor:
    num_steps_chinchilla_optimal = get_chinchilla_optimal_steps(model_size)
    return torch.tensor(steps) / num_steps_chinchilla_optimal


@save_beartype
def choose_steps_by_percentages(
        model_size: str,
        percentages: Sequence[float],
) -> Sequence[int]:
    """Choose steps by percentage of Chinchilla optimal."""
    percentage_of_chinchilla_optimal = get_percentage_of_chinchilla_optimal(model_size)
    steps = []
    for percentage in percentages:
        steps.append(STEPS[torch.argmin(torch.abs(percentage_of_chinchilla_optimal - percentage))].item())
    return steps


@save_beartype
def deduplicate_steps_and_percentages(
        steps: Sequence[int], 
        percentages: Sequence[float]
) -> tuple[Sequence[int], Sequence[float]]:
    steps_count = {}
    duplicate_idx = []
    for i, step in enumerate(steps):
        if step in steps_count.keys():
            steps_count[step] += 1
            duplicate_idx.append(i)
        else:
            steps_count[step] = 1

    for i in duplicate_idx[::-1]:
        steps.pop(i)
        percentages.pop(i)

    return steps, percentages


@save_beartype
def sparsify_band(
        tensor: torch.Tensor | nn.Parameter,
        band: tuple[float, float],
        taken_from: str,
        inplace: bool = False,
) -> tuple[torch.Tensor | nn.Parameter, int, int]:
    """Sparsify a tensor by setting values within a band to zero.

    Args:
        tensor: 
            The tensor (or Parameter) to sparsify.
        band: 
            The band of percentage of the values to sparsify. 
            For example, if band=(0.0, 0.1), the bottom 10% of the values will be set to zero.
            If, instead, band=(0.1, 0.2), the bottom 10% to 20% of the values will be set to zero.
        inplace: 
            Whether to modify the tensor in-place.
        taken_from:
            "pos", "neg", or "abs".

    Returns:
        The sparsified tensor (or Parameter).
    """
    if not inplace:
        tensor = tensor.clone()

    # Calculate the number of elements to sparsify
    num_elements = tensor.numel()
    start_idx = int(band[0] * num_elements)
    end_idx = int(band[1] * num_elements)

    # Flatten the tensor for simplicity
    flat_tensor = tensor.view(-1)

    if taken_from == "pos":
        sign = 1
    elif taken_from == "neg":
        sign = -1
        flat_tensor = -flat_tensor
    elif taken_from == "abs":
        sign = flat_tensor.sign()
        flat_tensor = flat_tensor.abs()
    else:
        raise ValueError(f"take_from={taken_from} not supported.")

    # Sort the tensor to find the threshold for sparsification
    values, indices = flat_tensor.sort()

    # We want the smallest absolute numbers with positive/negative sign;
    # however, in the sorted tensor, a bunch of large negative numbers will come before
    # the small positive numbers. So, we need to find the first zero in the sorted tensor
    # and then add that index to the start and end indices.
    if taken_from in ["pos", "neg"] and band[0] == 0.0:
        # because sign is switched for neg, always sparsify pos vals here
        first_zero_idx = torch.sum(values < 0).item()
        start_idx += first_zero_idx
        end_idx += first_zero_idx


    # Identify the indices to sparsify
    sparsify_indices = indices[start_idx:end_idx]

    num_sparsified_pos = int(torch.sum((sign * flat_tensor)[sparsify_indices] > 0).item())
    num_sparsified_neg = int(torch.sum((sign * flat_tensor)[sparsify_indices] < 0).item())

    # Set the specified values to zero
    flat_tensor[sparsify_indices] = 0

    # Reshape the tensor back to its original shape
    return (
        (sign * flat_tensor).view_as(tensor),
        num_sparsified_pos,
        num_sparsified_neg,
    )


@save_beartype
def sparsify_model(
        model: GPTNeoXForCausalLM,
        sparsity_band: tuple[float, float],
        taken_from: str,
        inplace: bool = False,
) -> tuple[
        GPTNeoXForCausalLM,  # sparsified model
        float,               # std
        float,               # mean   
        float,               # maximum
        float,               # minimum
        str,                 # max_param
        str,                 # min_param
        int,                 # numel
        int,                 # num_nonzero
        int,                 # num_sparsified_pos
        int                  # num_sparsified_neg
]:
    stds = []
    num_nonzero = 0
    mean = 0.0
    numel = 0
    maximum = -1e9
    minimum = 1e9
    num_sparsified_pos = 0 
    num_sparsified_neg = 0
    max_param = ""
    min_param = ""

    for name, parameter in model.named_parameters():
        if sparsity_band != (0.0, 0.0):
            parameter.data, npos, nneg = sparsify_band(
                parameter.data, 
                sparsity_band, 
                inplace=inplace,
                taken_from=taken_from,
            )
            num_sparsified_pos += npos
            num_sparsified_neg += nneg
        stds.append(parameter.data.std().item())

        new_maximum = parameter.data.max().item()
        new_minimum = parameter.data.min().item()

        if new_maximum > maximum:
            maximum = new_maximum
            max_param = name
        if new_minimum < minimum:
            minimum = new_minimum
            min_param = name
        
        num_nonzero += (parameter != 0.0).sum().item()
        mean += parameter.data.mean().item()
        numel += parameter.data.numel()

    std = np.mean(stds).item()
    mean = mean / numel

    return (
        model, std, mean, 
        maximum, minimum, 
        max_param, min_param,
        numel, num_nonzero, num_sparsified_pos, num_sparsified_neg,
    )


@save_beartype
def get_batches(
        input_ids: torch.Tensor,
        num_tokens_per_sample: int,
        batch_size: int,
) -> list[torch.Tensor]:
    """Get batches of input_ids."""
    batches = []
    input_ids = input_ids.reshape(-1)
    
    numels_batch = int(num_tokens_per_sample * batch_size)
    cutoff = len(input_ids) - len(input_ids) % numels_batch
    for i in range(0, cutoff, numels_batch):
        batch = input_ids[i:i+numels_batch]
        batch = batch.reshape(batch_size, -1)
        batches.append(batch)

    # Shuffle the batches
    np.random.shuffle(batches)
    return batches


@save_beartype
@save_inference_mode
def calculate_loss(
        model: GPTNeoXForCausalLM, 
        tokenizer: AutoTokenizer, 
        text: str | Sequence[str],
        device: str | int | torch.device,
        loop: tqdm,
        loop_description: str,
) -> torch.Tensor:
    """Evaluate the perplexity of a model on a text."""
    num_tokens_per_sample = 128
    batch_size = 32

    # Tokenize batch of texts
    tokens = tokenizer(text, return_tensors='pt')
    input_ids = tokens.input_ids.to(device)

    # Batchify the input_ids
    batches = get_batches(input_ids, num_tokens_per_sample, batch_size)

    # Calculate average loss for each text
    total_loss = 0.0

    for i, batch in enumerate(batches):
        loop.set_description(loop_description + f"; batch {i+1}/{len(batches)}")
        total_loss += model(batch, labels=batch).loss.mean().item()

    average_loss = total_loss / len(batches)

    return average_loss


@save_beartype
def load_test_datasets():
    raw_data_source = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'
    raw_data_cache = './data_raw/' # where to cache the data after downloading
    
    if not os.path.isfile(raw_data_cache):
        os.makedirs(raw_data_cache, exist_ok=True)
        urllib.request.urlretrieve(raw_data_source, raw_data_cache+'data.zip')

    if not os.path.isfile('data_raw/wikitext-103-raw/wiki.train.raw'):
        with zipfile.ZipFile(os.path.join(raw_data_cache, 'data.zip'), 'r') as zip_ref:
            zip_ref.extractall('data_raw/')

    with open('data_raw/wikitext-103-raw/wiki.train.raw', 'r', encoding="utf8") as data_file:
        train_data = data_file.read()

    with open('data_raw/wikitext-103-raw/wiki.valid.raw', 'r', encoding="utf8") as data_file:
        eval_data = data_file.read()

    return train_data, eval_data


@save_beartype
def main() -> None:
    dtype = torch.bfloat16  # hopefully fits on machine
    percentages = [0.25, 0.5, 0.8, 0.9, 1.0]
    additional_steps = [i*1000 for i in range(10, 144, 10)] + [143_000]
    sparsity_bands = [
        # Provide baseline
        (0.0, 0.0),

        # Sparsify the bottom n% of the parameters
        (0.0, 1e-4),
        (0.0, 1e-3),
        (0.0, 1e-2),
        (0.0, 5e-2),
        (0.0, 1e-1),

        # Sparsify the top n% of the parameters
        (1-1e-4, 1.0),
        (1-1e-3, 1.0),
        (1-1e-2, 1.0),
        (1-5e-2, 1.0),
        (1-1e-1, 1.0),

        # Sparsify the middle n% of the parameters as a control group
        (0.5-5e-5, 0.5+5e-5),
        (0.5-5e-4, 0.5+5e-4),
        (0.5-5e-3, 0.5+5e-3),
        (0.5-2.5e-2, 0.5+2.5e-2),
        (0.5-5e-2, 0.5+5e-2),
    ]
    _, dataset = load_test_datasets()

    for model_size in MODEL_SIZES:
        steps = choose_steps_by_percentages(model_size, percentages)
        additional_percentages = get_percentage_of_chinchilla_optimal(model_size, additional_steps)
        steps += additional_steps
        crnt_percentages = percentages + additional_percentages.tolist()
        steps, crnt_percentages = deduplicate_steps_and_percentages(steps, crnt_percentages)
        taken_froms = ["pos", "neg", "abs"]

        for step_idx, step in enumerate(steps):
            rich.print(
                f"\n\nModel: {model_size}, "
                f"Step: {step}, "
                f"Chinchilla: {crnt_percentages[step_idx] * 100:.2f}%, "
                f"Iteration: {step_idx+1}/{len(steps)}\n\n"
            )

            results = {
                "step": [],
                "percentage_chinchilla_optimal": [],
                "sparsity_band": [],
                "taken_from": [],
                "loss": [],
                "std": [],
                "maximum": [],
                "minimum": [],
                "max_param": [],
                "min_param": [],
                "mean": [],
                "numel": [],
                "num_nonzero": [],
                "num_sparsified_pos": [],
                "num_sparsified_neg": [],
            }

            tokenizer = AutoTokenizer.from_pretrained(
                f"EleutherAI/pythia-{model_size}",
                revision=f"step{step}",
                cache_dir=f"./models/pythia-{model_size}/step{step}",
            )

            loop = tqdm(
                itertools.product(sparsity_bands, taken_froms), 
                total=len(sparsity_bands) * len(taken_froms),
            )
            for sparsity_band, taken_from in loop:
                if (sparsity_band == (0.0, 0.0)) and (taken_from != "pos"):
                    continue  # only calc baseline once

                description = f"{sparsity_band=}"
                loop.set_description(description + "; preparing model")

                # Reload model every time and then sparsify (to avoid accumulating sparsity)
                model = GPTNeoXForCausalLM.from_pretrained(
                    f"EleutherAI/pythia-{model_size}",
                    revision=f"step{step}",
                    cache_dir=f"./models/pythia-{model_size}/step{step}",
                )

                model = model.to(device="cuda", dtype=dtype)
                model.eval()
                model.requires_grad_(False)

                (
                    model, std, mean, 
                    maximum, minimum, 
                    max_param, min_param,
                    numel, num_nonzero, num_sparsified_pos, num_sparsified_neg,
                ) = sparsify_model(
                    model=model, 
                    sparsity_band=sparsity_band, 
                    taken_from=taken_from,
                    inplace=False, 
                )

                loss = calculate_loss(
                    model, tokenizer, dataset, "cuda", loop, description,
                )
                loop.write(
                    f"{sparsity_band}, {taken_from}, "
                    f"loss={loss:.3f}, "
                    f"{std=:.3f}, max={maximum:.3f}, min={minimum:.3f}, "
                    f"{numel=:,}, non0={num_nonzero:,}, "
                    f"nsp={num_sparsified_pos:,}, nsn={num_sparsified_neg:,}"
                )

                results["step"].append(step)
                results["percentage_chinchilla_optimal"].append(crnt_percentages[step_idx])
                results["sparsity_band"].append(sparsity_band)
                results["taken_from"].append(taken_from)
                results["loss"].append(loss)
                results["std"].append(std)
                results["mean"].append(mean)
                results["maximum"].append(maximum)
                results["minimum"].append(minimum)
                results["max_param"].append(max_param)
                results["min_param"].append(min_param)
                results["numel"].append(numel)
                results["num_nonzero"].append(num_nonzero)
                results["num_sparsified_pos"].append(num_sparsified_pos)
                results["num_sparsified_neg"].append(num_sparsified_neg)

            shutil.rmtree(f"./models/pythia-{model_size}/step{step}")

            results = pd.DataFrame(results)
            results_dir = "results/sparsified"
            os.makedirs(results_dir, exist_ok=True)
            with open(f"{results_dir}/pythia-{model_size}.csv", "a") as f:
                results.to_csv(f, header=f.tell()==0, index=False)


if __name__ == "__main__":
    main()
