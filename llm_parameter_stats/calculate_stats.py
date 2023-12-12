from __future__ import annotations

import sys
import functools

import os
import shutil
import itertools
from tqdm import tqdm
from time import perf_counter

import rich
try:
    from beartype import beartype
except ImportError:
    pass
import torch
from torch import nn
import pandas as pd
import numpy as np
from transformers import GPTNeoXForCausalLM
from packaging import version


#####################
# --- CONSTANTS --- #
#####################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_HISTOGRAM_BINS = 100

MODEL_SIZES = [
        # "70m", "70m-deduped",
        # "160m", "160m-deduped",
        # "410m", "410m-deduped", 
        "1b", "1b-deduped",
        "1.4b", "1.4b-deduped",
        "2.8b", "2.8b-deduped",
        # "6.9b", "6.9b-deduped",
        # "12b", "12b-deduped",
]
STEPS_POWER_OF_TWO = [2**i for i in range(10)]
STEPS = [0] + STEPS_POWER_OF_TWO + [i * 1000 for i in range(1, 144)]


###################
# --- HELPERS --- #
###################


def save_beartype(func):
    if version.parse(sys.version.split(" ")[0]) >= version.parse("3.10"):
        return beartype(func)
    else:
        return func


def save_inference_mode(func):
    if hasattr(torch, "inference_mode") and callable(torch.inference_mode):
        return torch.inference_mode()(func)
    else:
        return torch.no_grad()(func)


def pairwise(x):
    try:
        return itertools.pairwise(x)
    except AttributeError:
        return zip(x[:-1], x[1:])

@save_beartype
def to_python(x: torch.Tensor | nn.Parameter | float) -> float | list[float]:
    if isinstance(x, float):
        return x
    if x.numel() == 1:
        return x.item()
    return x.detach().cpu().tolist()


@save_beartype
def to_numpy(x: torch.Tensor | nn.Parameter | float) -> np.ndarray:
    if isinstance(x, float):
        return np.array(x)
    return x.detach().cpu().numpy()


@save_beartype
def load_model(model_size: str, step: int, cache_dir: str) -> GPTNeoXForCausalLM:
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}",
        revision=f"step{step}",
        cache_dir=cache_dir,
    )
    return model


@save_beartype
def optional_remove(model_size: str, current_step: int) -> None:
    """
    Remove steps that aren't needed anymore, 
    but keep them for at least 10000 steps to not have to download another model.
    """
    if current_step in STEPS_POWER_OF_TWO:
        shutil.rmtree(f"models/pythia-{model_size}/step{current_step}")
    if current_step >= 11_000:
        shutil.rmtree(f"models/pythia-{model_size}/step{current_step - 11_000}")


@save_beartype
def initialize_results_dicts() -> tuple[dict[str, ist[float | str]], ...]:
    results_intra_parameter = {
        "parameter": [],
        "step": [],
        "mean": [],
        "median": [],
        "mode": [],
        "std": [],
        "skewness": [],
        "kurtosis": [],
        "maximum": [],
        "minimum": [],
        "abs_mean": [],
        "abs_std": [],
        "abs_maximum": [],
        "abs_minimum": [],
        "eighty_percentile": [],
        "ninety_percentile": [],
        "ninety_five_percentile": [],
        "ninety_nine_percentile": [],
        "sparsity_0": [],
        "sparsity_1e-6": [],
        "sparsity_1e-5": [],
        "sparsity_1e-4": [],
        "sparsity_1e-3": [],
        "sparsity_1e-2": [],
        "sparsity_1e-1": [],
        "L1": [],
        "L2": [],
    }

    results_histogram = {
        "parameter": [],
        "step": [],
        "counts": [],
        "bin_centers": [],
        "bin_width": [],
    }

    results_inter_parameter = {
        "parameter": [],
        "step": [],
        "step_next": [],
        "cos_sim": [],
        "l1_change": [],
        "l2_change": [],
        "realtive_change": [],
        "mean_squared_change": [],
        "max_abs_change": [],
    }

    return results_intra_parameter, results_histogram, results_inter_parameter


@save_beartype
def initialize_accumulated_parameter_dict() -> dict[str, torch.Tensor]:
    parameter_dict = {
        "all_parameters": torch.tensor([]),
        "all_weights": torch.tensor([]),
        "all_biases": torch.tensor([]),
        "all_dense_weights": torch.tensor([]),
        "all_dense_biases": torch.tensor([]),
        "all_attention_weights": torch.tensor([]),
        "all_attention_biases": torch.tensor([]),
        "all_layernorm_weights": torch.tensor([]),
        "all_layernorm_biases": torch.tensor([]),
    }
    return parameter_dict


@save_beartype
def accumulate_parameters(
        name: str, 
        parameter: nn.Parameter | torch.Tensor,
        parameter_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    parameter_dict["all_parameters"] = torch.cat((parameter_dict["all_parameters"], parameter.flatten()))
    if "weight" in name:
        parameter_dict["all_weights"] = torch.cat((parameter_dict["all_weights"], parameter.flatten()))
        if "layernorm" in name:
            parameter_dict["all_layernorm_weights"] = torch.cat((parameter_dict["all_layernorm_weights"], parameter.flatten()))
        elif "attention" in name:
            parameter_dict["all_attention_weights"] = torch.cat((parameter_dict["all_attention_weights"], parameter.flatten()))
        else:
            parameter_dict["all_dense_weights"] = torch.cat((parameter_dict["all_dense_weights"], parameter.flatten()))
    elif "bias" in name:
        parameter_dict["all_biases"] = torch.cat((parameter_dict["all_biases"], parameter.flatten()))
        if "layernorm" in name:
            parameter_dict["all_layernorm_biases"] = torch.cat((parameter_dict["all_layernorm_biases"], parameter.flatten()))
        elif "attention" in name:
            parameter_dict["all_attention_biases"] = torch.cat((parameter_dict["all_attention_biases"], parameter.flatten()))
        else:
            parameter_dict["all_dense_biases"] = torch.cat((parameter_dict["all_dense_biases"], parameter.flatten()))
    return parameter_dict
    


@save_beartype
def get_title(model_size: str) -> str:
    title = "| ANALYZING NEW MODEL SIZE |"
    width = len(title)
    title = f"\n\n{'=' * width}\n{title}\n{'-' * width}\n"

    size = f"| Size: {model_size} "
    spaces = " " * (width - len(size) - 1)
    title += f"{size}{spaces}|\n{'=' * width}\n\n"
    return title


@save_beartype
def to_hours_minutes_seconds(time_seconds: float) -> tuple[int, int, int]:
    hours, remainder = divmod(time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(round(hours)), int(round(minutes)), int(round(seconds))


@save_beartype
def time_passed(start_time: float) -> tuple[int, int, int]:
    time_seconds = perf_counter() - start_time
    return to_hours_minutes_seconds(time_seconds)


######################
# --- STATISTICS --- #
######################

@save_beartype
def calculate_sparsity(
        tensor: torch.Tensor | nn.Parameter, 
        threshold: float = 0.0
) -> float:
    """
    Calculate the sparsity of a tensor.

    Parameters:
    tensor (torch.Tensor | nn.Parameter): The input tensor.
    threshold (float): Values below this threshold are considered sparse. Default is 0.0.

    Returns:
    float: The sparsity of the tensor.
    """
    # By default, the sparsity is calculated based on zero values.
    if threshold == 0:
        sparse_elements = torch.sum(tensor == 0).item()
    else:
        sparse_elements = torch.sum(torch.abs(tensor) < threshold).item()

    total_elements = tensor.numel()
    sparsity = sparse_elements / total_elements
    return sparsity


@save_beartype
def skewness(tensor: torch.Tensor | nn.Parameter) -> float:
    tensor = tensor.flatten().float()
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    skewness = torch.mean((tensor - mean) ** 3) / std ** 3
    return skewness.item()


@save_beartype
def kurtosis(tensor: torch.Tensor | nn.Parameter) -> float:
    tensor = tensor.flatten().float()
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    kurtosis = torch.mean((tensor - mean) ** 4) / std ** 4 - 3
    return kurtosis.item()


#####################################
# --- ADD STATISTICS TO RESULTS --- #
#####################################

@save_beartype
def add_intra_parameter_statistics(
        results: dict[str, str | float],
        parameter: nn.Parameter | torch.Tensor,
        name: str, 
        step: int,
) -> dict[str, str | float]:
    results["parameter"].append(name)
    results["step"].append(step)

    # Calculate standard statistics
    results["mean"].append(to_python(parameter.mean()))
    results["median"].append(to_python(parameter.median()))
    results["mode"].append(to_python(parameter.flatten().mode().values))
    results["std"].append(to_python(parameter.std()))
    results["skewness"].append(skewness(parameter))
    results["kurtosis"].append(kurtosis(parameter))
    results["maximum"].append(to_python(torch.max(parameter)))
    results["minimum"].append(to_python(parameter.min()))

    # The same for the absolute values
    #   This is useful for finding the skewness of the distribution.
    results["abs_mean"].append(to_python(parameter.abs().mean()))
    results["abs_std"].append(to_python(parameter.abs().std()))
    results["abs_maximum"].append(to_python(parameter.abs().max()))
    results["abs_minimum"].append(to_python(parameter.abs().min()))

    # Calculate the percentiles. 
    #  Unfortunately, torch cannot handle large tensors, so use numpy instead.
    results["eighty_percentile"].append(np.quantile(to_numpy(parameter), 0.8).tolist())
    results["ninety_percentile"].append(np.quantile(to_numpy(parameter), 0.9).tolist())
    results["ninety_five_percentile"].append(np.quantile(to_numpy(parameter), 0.95).tolist())
    results["ninety_nine_percentile"].append(np.quantile(to_numpy(parameter), 0.99).tolist())

    # Calculate the sparsity
    results["sparsity_0"].append(calculate_sparsity(parameter, 0.0))
    results["sparsity_1e-6"].append(calculate_sparsity(parameter, 1e-6))
    results["sparsity_1e-5"].append(calculate_sparsity(parameter, 1e-5))
    results["sparsity_1e-4"].append(calculate_sparsity(parameter, 1e-4))
    results["sparsity_1e-3"].append(calculate_sparsity(parameter, 1e-3))
    results["sparsity_1e-2"].append(calculate_sparsity(parameter, 1e-2))
    results["sparsity_1e-1"].append(calculate_sparsity(parameter, 1e-1))

    # Calculate the L1 and L2 norms
    results["L1"].append(to_python(parameter.norm(1)))
    results["L2"].append(to_python(parameter.norm(2)))

    return results


@save_beartype
def add_inter_parameter_statistics(
        results: dict[str, str | float],
        parameter_now: nn.Parameter | torch.Tensor,
        parameter_last: nn.Parameter | torch.Tensor,
        name: str,
        step_last: int,
        step_now: int,
) -> dict[str, str | float]:
    results["parameter"].append(name)
    results["step"].append(step_last)
    results["step_next"].append(step_now)

    cos_sim = torch.nn.functional.cosine_similarity(parameter_now.flatten(), parameter_last.flatten(), dim=0)
    results["cos_sim"].append(cos_sim.detach().cpu().numpy())

    delta = parameter_now - parameter_last
    results["l1_change"].append(to_python(delta.norm(1)))
    results["l2_change"].append(to_python(delta.norm(2)))
    results["realtive_change"].append(
        to_python(delta.norm(2) / parameter_last.norm(2)) 
        if parameter_last.norm(2) != 0 
        else 0.0
    )
    results["mean_squared_change"].append(to_python(torch.mean(delta ** 2)))
    results["max_abs_change"].append(to_python(torch.max(delta.abs())))

    return results


@save_beartype
def add_histogram(
        results: dict[str, str | float],
        parameter: nn.Parameter | torch.Tensor,
        name: str,
        step: int,
) -> dict[str, str | float]:
    parameter = parameter.flatten().detach().cpu()
    counts, bin_edges = torch.histogram(parameter, bins=NUM_HISTOGRAM_BINS)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    results["parameter"].append(name)
    results["step"].append(step)
    results["counts"].append(to_python(counts))
    results["bin_centers"].append(to_python(bin_centers))
    results["bin_width"].append(bin_width.item())

    return results


@save_beartype
def add_accumulated_parameter_statistics(
        intra_parameter_dict: dict[str, list[str | float]],
        histogram_dict: dict[str, list[str | float]],
        parameter_dict: dict[str, torch.Tensor],
        step_n: int,
) -> tuple[dict[str, list[str | float]], dict[str, list[str | float]]]:
    for name, parameter in parameter_dict.items():
        if len(parameter) == 0:
            continue
        intra_parameter_dict = add_intra_parameter_statistics(
            intra_parameter_dict, parameter, name, step_n
        )
        histogram_dict = add_histogram(
            histogram_dict, parameter, name, step_n
        )
    return intra_parameter_dict, histogram_dict


@save_beartype
def add_accumulated_inter_parameter_statistics(
        inter_parameter_dict: dict[str, list[str | float]],
        parameter_dict: dict[str, torch.Tensor],
        parameter_dict_next: dict[str, torch.Tensor],
        step_n: int,
        step_n_next: int,
) -> dict[str, list[str | float]]:
    for name, parameter in parameter_dict.items():
        if len(parameter) == 0:
            continue
        inter_parameter_dict = add_inter_parameter_statistics(
            inter_parameter_dict, parameter, parameter_dict_next[name], name, step_n, step_n_next
        )
    return inter_parameter_dict


@save_beartype
def accumulate_and_calculate_parameter_group_stats(
        results_histogram: dict[str, list[str | float]],
        results_intra_parameter: dict[str, list[str | float]],
        results_inter_parameter: dict[str, list[str | float]],
        model_n: GPTNeoXForCausalLM,
        model_n_next: GPTNeoXForCausalLM,
        steps: list[int],
        step_n: int,
        step_n_next: int,
        num_parameters: int,
        groups: list[str],
        inter_parameter_only: bool = False,
) -> tuple[dict[str, list[str | float]], dict[str, list[str | float]]]:
    """Do this grop by group to not duplicate the parameters in memory."""
    # Accumulate the parameters
    parameter_dict = initialize_accumulated_parameter_dict()
    parameter_dict_next = initialize_accumulated_parameter_dict()

    rich.print(f"Accumulating parameters for {', '.join(groups)}...")
    for (name_n, parameter_n), (name_n_next, parameter_n_next) in tqdm(
        zip(model_n.named_parameters(), model_n_next.named_parameters()),
        total=num_parameters,
    ):
        for parameter_group in groups:
            if parameter_group == "all_parameters":
                parameter_dict[parameter_group] = torch.cat((parameter_dict[parameter_group], parameter_n.flatten()))
                parameter_dict_next[parameter_group] = torch.cat((parameter_dict_next[parameter_group], parameter_n_next.flatten()))
            elif parameter_group == "all_weights":
                if "weight" in name_n:
                    parameter_dict[parameter_group] = torch.cat((parameter_dict[parameter_group], parameter_n.flatten()))
                    parameter_dict_next[parameter_group] = torch.cat((parameter_dict_next[parameter_group], parameter_n_next.flatten()))
            elif parameter_group == "all_biases":
                if "bias" in name_n:
                    parameter_dict[parameter_group] = torch.cat((parameter_dict[parameter_group], parameter_n.flatten()))
                    parameter_dict_next[parameter_group] = torch.cat((parameter_dict_next[parameter_group], parameter_n_next.flatten()))
            elif parameter_group == "all_dense_weights":
                if "weight" in name_n and "layernorm" not in name_n and "attention" not in name_n:
                    parameter_dict[parameter_group] = torch.cat((parameter_dict[parameter_group], parameter_n.flatten()))
                    parameter_dict_next[parameter_group] = torch.cat((parameter_dict_next[parameter_group], parameter_n_next.flatten()))
            elif parameter_group == "all_dense_biases":
                if "bias" in name_n and "layernorm" not in name_n and "attention" not in name_n:
                    parameter_dict[parameter_group] = torch.cat((parameter_dict[parameter_group], parameter_n.flatten()))
                    parameter_dict_next[parameter_group] = torch.cat((parameter_dict_next[parameter_group], parameter_n_next.flatten()))
            elif parameter_group == "all_attention_weights":
                if "weight" in name_n and "attention" in name_n:
                    parameter_dict[parameter_group] = torch.cat((parameter_dict[parameter_group], parameter_n.flatten()))
                    parameter_dict_next[parameter_group] = torch.cat((parameter_dict_next[parameter_group], parameter_n_next.flatten()))
            elif parameter_group == "all_attention_biases":
                if "bias" in name_n and "attention" in name_n:
                    parameter_dict[parameter_group] = torch.cat((parameter_dict[parameter_group], parameter_n.flatten()))
                    parameter_dict_next[parameter_group] = torch.cat((parameter_dict_next[parameter_group], parameter_n_next.flatten()))
            elif parameter_group == "all_layernorm_weights":
                if "weight" in name_n and "layernorm" in name_n:
                    parameter_dict[parameter_group] = torch.cat((parameter_dict[parameter_group], parameter_n.flatten()))
                    parameter_dict_next[parameter_group] = torch.cat((parameter_dict_next[parameter_group], parameter_n_next.flatten()))
            elif parameter_group == "all_layernorm_biases":
                if "bias" in name_n and "layernorm" in name_n:
                    parameter_dict[parameter_group] = torch.cat((parameter_dict[parameter_group], parameter_n.flatten()))
                    parameter_dict_next[parameter_group] = torch.cat((parameter_dict_next[parameter_group], parameter_n_next.flatten()))

    # Calculate the statistics
    rich.print(f"Calculating statistics for {', '.join(groups)}...")
    if not inter_parameter_only:
        results_intra_parameter, results_histogram = add_accumulated_parameter_statistics(
            results_intra_parameter, results_histogram, parameter_dict, step_n
        )
        if step_n_next == steps[-1]:
            results_intra_parameter, results_histogram = add_accumulated_parameter_statistics(
                results_intra_parameter, results_histogram, parameter_dict_next, step_n_next
            )
    results_inter_parameter = add_accumulated_inter_parameter_statistics(
        results_inter_parameter, parameter_dict, parameter_dict_next, step_n, step_n_next
    )

    # Free up memory
    del parameter_dict, parameter_dict_next

    return results_intra_parameter, results_histogram, results_inter_parameter


################
# --- MAIN --- #
################

@save_beartype
@save_inference_mode
def main() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    steps = STEPS
    model_sizes =  MODEL_SIZES

    for model_size in model_sizes:
        print(get_title(model_size))

        start_time_model = perf_counter()
        results_intra_parameter, results_histogram, results_inter_parameter = initialize_results_dicts()
        model_n = load_model(model_size, steps[0], f"models/pythia-{model_size}/step{steps[0]}")
        model_n_next = load_model(model_size, steps[1], f"models/pythia-{model_size}/step{steps[1]}")

        # Store errors in list to save them later
        # Don't save errors in above model initilization, because I want to see problems with it
        #   immediately, instead of after hours of waiting.
        errors = []

        num_parameters = 0
        for _, _ in model_n.named_parameters():
            num_parameters += 1
            del _ 

        for i, (step_n, step_n_next) in enumerate(pairwise(steps)):
            start_time_step = perf_counter()

            rich.print(f"\n{model_size}: {step_n=}, {step_n_next=} :: number {i+1}/{len(steps)-1}\n")

            cache_dir_last = f"models/pythia-{model_size}/step{step_n}"
            cache_dir = f"models/pythia-{model_size}/step{step_n_next}"

            # Load the models
            try:
                model_n = load_model(model_size, step_n, cache_dir_last) if step_n == 0 else model_n_next
                model_n_next = load_model(model_size, step_n_next, cache_dir)
            except EnvironmentError as e:
                errors.append(repr(e))
                rich.print(f"ERROR: {e}")
                continue 
            except OSError as e:
                errors.append(repr(e))
                rich.print(f"ERROR: {e}")
                continue

            # Calculate the statistics
            rich.print("Calculating statistics...")
            for (name_n, parameter_n), (name_n_next, parameter_n_next) in tqdm(
                zip(model_n.named_parameters(), model_n_next.named_parameters()),
                total=num_parameters,
            ):
                # Intra-parameter statistics
                parameter_n = parameter_n.to(DEVICE)
                results_intra_parameter = add_intra_parameter_statistics(results_intra_parameter, parameter_n, name_n, step_n)
                parameter_n = parameter_n.to("cpu")  # make sure to free up memory

                results_histogram = add_histogram(results_histogram, parameter_n, name_n, step_n)

                if step_n_next == steps[-1]:
                    parameter_n_next = parameter_n_next.to(DEVICE)  # speed up calculations
                    results_intra_parameter = add_intra_parameter_statistics(results_intra_parameter, parameter_n_next, name_n_next, step_n_next)
                    parameter_n_next = parameter_n_next.to("cpu")  # free up memory
                    results_histogram = add_histogram(results_histogram, parameter_n_next, name_n_next, step_n_next)

                # Inter-parameter statistics
                parameter_n = parameter_n.to(DEVICE)
                parameter_n_next = parameter_n_next.to(DEVICE)
                results_inter_parameter = add_inter_parameter_statistics(
                    results_inter_parameter, parameter_n, parameter_n_next, name_n, step_n, step_n_next
                )
                parameter_n = parameter_n.to("cpu")
                parameter_n_next = parameter_n_next.to("cpu")

            # Calculate for all parameters
            acc_and_calc = functools.partial(
                accumulate_and_calculate_parameter_group_stats,
                results_histogram=results_histogram,
                results_intra_parameter=results_intra_parameter,
                results_inter_parameter=results_inter_parameter,
                model_n=model_n,
                model_n_next=model_n_next,
                steps=steps,
                step_n=step_n,
                step_n_next=step_n_next,
                num_parameters=num_parameters,
            )

            results_intra_parameter, results_histogram, results_inter_parameter = acc_and_calc(
                groups=["all_parameters"]
            )
            results_intra_parameter, results_histogram, results_inter_parameter = acc_and_calc(
                groups=["all_weights", "all_biases"]
            )
            results_intra_parameter, results_histogram, results_inter_parameter = acc_and_calc(
                groups=[
                    "all_dense_weights", "all_dense_biases", 
                    "all_attention_weights", "all_attention_biases", 
                    "all_layernorm_weights", "all_layernorm_biases",
                ]
            )


            # Calculate inter-parameter statistics in 10_000-step-intervals
            if step_n_next >= 10_000:
                rich.print("Calculating statistics for 10_000-step-intervals...")

                cache_dir_10_000 = f"models/pythia-{model_size}/step{step_n_next - 10_000}"
                model_10_000 = load_model(model_size, step_n_next - 10_000, cache_dir_10_000)

                for (name_n, parameter_n), (name_n_next, parameter_n_next) in tqdm(
                    zip(model_10_000.named_parameters(), model_n_next.named_parameters()),
                    total=num_parameters,
                ):
                    # Inter-parameter statistics
                    parameter_n = parameter_n.to(DEVICE)
                    parameter_n_next = parameter_n_next.to(DEVICE)
                    results_inter_parameter = add_inter_parameter_statistics(
                        results_inter_parameter, parameter_n, parameter_n_next, name_n, step_n_next - 10_000, step_n_next
                    )
                    parameter_n = parameter_n.to("cpu")
                    parameter_n_next = parameter_n_next.to("cpu")

                # Add data for all parameters
                acc_and_calc = functools.partial(
                    accumulate_and_calculate_parameter_group_stats,
                    results_histogram=results_histogram,
                    results_intra_parameter=results_intra_parameter,
                    results_inter_parameter=results_inter_parameter,
                    model_n=model_10_000,
                    model_n_next=model_n_next,
                    steps=steps,
                    step_n=step_n_next - 10_000,
                    step_n_next=step_n_next,
                    num_parameters=num_parameters,
                    inter_parameter_only=True,
                )

                results_intra_parameter, results_histogram, results_inter_parameter = acc_and_calc(
                    groups=["all_parameters"]
                )
                results_intra_parameter, results_histogram, results_inter_parameter = acc_and_calc(
                    groups=["all_weights", "all_biases"]
                )
                results_intra_parameter, results_histogram, results_inter_parameter = acc_and_calc(
                    groups=[
                        "all_dense_weights", "all_dense_biases", 
                        "all_attention_weights", "all_attention_biases", 
                        "all_layernorm_weights", "all_layernorm_biases",
                    ]
                )

                # Free up storage
                optional_remove(model_size, step_n)

            # Save all the results
            os.makedirs(f"results/pythia-{model_size}", exist_ok=True)

            df_intra_parameter = pd.DataFrame(results_intra_parameter)
            df_inter_parameter = pd.DataFrame(results_inter_parameter)
            df_histogram = pd.DataFrame(results_histogram)

            with open(f"results/pythia-{model_size}/intra_parameter.csv", 'a') as f:
                df_intra_parameter.to_csv(f, header=f.tell()==0, index=False)
            with open(f"results/pythia-{model_size}/inter_parameter.csv", 'a') as f:
                df_inter_parameter.to_csv(f, header=f.tell()==0, index=False)
            with open(f"results/pythia-{model_size}/histogram.csv", 'a') as f:
                df_histogram.to_csv(f, header=f.tell()==0, index=False)

            # Re-initialize the result-dirs
            results_intra_parameter, results_histogram, results_inter_parameter = initialize_results_dicts()

            # Print the time it took to calculate the statistics
            hours, minutes, seconds = time_passed(start_time_step)
            rich.print(f"\n\nTotal time for this step: {hours}:{minutes}:{seconds} (hrs:min:sec)")
            eta = (perf_counter() - start_time_step) * (len(steps) - i - 1)
            hours, minutes, seconds = to_hours_minutes_seconds(eta)
            rich.print(f"Estimated time remaining: {hours}:{minutes}:{seconds} (hrs:min:sec)\n\n")
            

        # Free up more memory
        shutil.rmtree(cache_dir)

        # Save the errors
        with open(f"results/pythia-{model_size}/errors.txt", 'w') as f:
            f.write("\n\n".join(errors))

        # Print the time it took to calculate the statistics
        hours, minutes, seconds = time_passed(start_time_model)
        rich.print(f"\n\nTotal time for this model: {hours}:{minutes}:{seconds} (hrs:min:sec)\n\n")



#######################
# --- LET'S GO!!! --- #
#######################

if __name__ == "__main__":
    main()
