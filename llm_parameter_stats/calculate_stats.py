from __future__ import annotations

import os
import shutil
import itertools
from tqdm import tqdm

import rich
try:
    from beartype import beartype_this_package
    beartype_this_package()
except ImportError:
    pass
import torch
from torch import nn
import pandas as pd
import numpy as np
from transformers import GPTNeoXForCausalLM


def pairwise(x):
    try:
        return itertools.pairwise(x)
    except AttributeError:
        return zip(x[:-1], x[1:])


#####################
# --- CONSTANTS --- #
#####################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_HISTOGRAM_BINS = 100

MODEL_SIZES = [
        "70m", "70m-deduped",
        "160m", "160m-deduped",
        "410m", "410m-deduped", 
        "1b", "1b-deduped",
        "1.4b", "1.4b-deduped",
        "2.8b", "2.8b-deduped",
        "6.9b", "6.9b-deduped",
        "12b", "12b-deduped",
]
STEPS = [0] + [2**i for i in range(10)] + [i * 1000 for i in range(1, 144)]


###################
# --- HELPERS --- #
###################

def to_python(x: torch.Tensor | nn.Parameter | float) -> float | list[float]:
    if isinstance(x, float):
        return x
    if x.numel() == 1:
        return x.item()
    return x.detach().cpu().tolist()


def to_numpy(x: torch.Tensor | nn.Parameter | float) -> np.ndarray:
    if isinstance(x, float):
        return np.array(x)
    return x.detach().cpu().numpy()


def load_model(model_size: str, step: int, cache_dir: str) -> GPTNeoXForCausalLM:
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}",
        revision=f"step{step}",
        cache_dir=cache_dir,
    )
    return model


def initialize_results_dicts() -> tuple[dict[str, str | float], ...]:
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


def get_title(model_size: str) -> str:
    title = "| ANALYZING NEW MODEL SIZE |"
    width = len(title)
    title = f"\n\n{'=' * width}\n{title}\n{'-' * width}\n"

    size = f"| Size: {model_size} "
    spaces = " " * (width - len(size) - 1)
    title += f"{size}{spaces}|\n{'=' * width}\n\n"
    return title


######################
# --- STATISTICS --- #
######################

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


def skewness(tensor: torch.Tensor | nn.Parameter) -> float:
    tensor = tensor.flatten().float()
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    skewness = torch.mean((tensor - mean) ** 3) / std ** 3
    return skewness.item()


def kurtosis(tensor: torch.Tensor | nn.Parameter) -> float:
    tensor = tensor.flatten().float()
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    kurtosis = torch.mean((tensor - mean) ** 4) / std ** 4 - 3
    return kurtosis.item()


#####################################
# --- ADD STATISTICS TO RESULTS --- #
#####################################

def add_intra_parameter_statistics(
        results: dict[str, str | float],
        parameter: nn.Parameter | torch.Tensor,
        name: str, 
        step: int,
) -> dict[str, str | float]:
    parameter = parameter.to(DEVICE)
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

    parameter = parameter.to("cpu")

    return results


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


def add_histogram(
        results: dict[str, str | float],
        parameter: nn.Parameter | torch.Tensor,
        name: str,
        step: int,
) -> dict[str, str | float]:
    counts, bin_edges = torch.histogram(parameter, bins=NUM_HISTOGRAM_BINS)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    results["parameter"].append(name)
    results["step"].append(step)
    results["counts"].append(to_python(counts))
    results["bin_centers"].append(to_python(bin_centers))
    results["bin_width"].append(bin_width.item())

    return results


################
# --- MAIN --- #
################

def main() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    steps = [0, 1, 2]  # STEPS
    model_sizes = ["70m"]#, "70m-deduped"]  # MODEL_SIZES

    for model_size in model_sizes:
        print(get_title(model_size))
        results_intra_parameter, results_histogram, results_inter_parameter = initialize_results_dicts()
        model_n = load_model(model_size, steps[0], f"models/pythia-{model_size}/step{steps[0]}")
        model_n_next = load_model(model_size, steps[1], f"models/pythia-{model_size}/step{steps[1]}")

        for i, (step_n, step_n_next) in enumerate(pairwise(steps)):
            rich.print(f"\nStep: {step_n=}, {step_n_next=} :: number {i+1}/{len(steps)-1}\n")

            cache_dir_last = f"models/pythia-{model_size}/step{step_n}"
            cache_dir = f"models/pythia-{model_size}/step{step_n_next}"

            # Load the models
            model_n = load_model(model_size, step_n, cache_dir_last) if step_n == 0 else model_n_next
            model_n_next = load_model(model_size, step_n_next, cache_dir)

            all_parameter_values = torch.tensor([])
            all_parameter_values_next = torch.tensor([])

            # Calculate the statistics
            for (name_n, parameter_n), (name_n_next, parameter_n_next) in zip(
                model_n.named_parameters(), model_n_next.named_parameters()
            ):
                # Intra-parameter statistics
                results_intra_parameter = add_intra_parameter_statistics(results_intra_parameter, parameter_n, name_n, step_n)
                all_parameter_values = torch.cat((all_parameter_values, parameter_n.flatten()))
                all_parameter_values_next = torch.cat((all_parameter_values_next, parameter_n_next.flatten()))
                results_histogram = add_histogram(results_histogram, parameter_n, name_n, step_n)
                if step_n_next == steps[-1]:
                    results_intra_parameter = add_intra_parameter_statistics(results_intra_parameter, parameter_n_next, name_n_next, step_n_next)
                    results_histogram = add_histogram(results_histogram, parameter_n_next, name_n_next, step_n_next)

                # Inter-parameter statistics
                results_inter_parameter = add_inter_parameter_statistics(
                    results_inter_parameter, parameter_n, parameter_n_next, name_n, step_n, step_n_next
                )

            # Add data for all parameters
            results_intra_parameter = add_intra_parameter_statistics(results_intra_parameter, all_parameter_values, "all_parameters", step_n)
            results_histogram = add_histogram(results_histogram, all_parameter_values, "all_parameters", step_n)
            results_inter_parameter = add_inter_parameter_statistics(
                results_inter_parameter, all_parameter_values, all_parameter_values_next, "all_parameters", step_n, step_n_next
            )
            if step_n_next == steps[-1]:
                results_intra_parameter = add_intra_parameter_statistics(
                    results_intra_parameter, all_parameter_values_next, "all_parameters", step_n_next
                )
                results_histogram = add_histogram(
                    results_histogram, all_parameter_values_next, "all_parameters", step_n_next
                )

            # Free up storage
            shutil.rmtree(cache_dir_last)

            # Calculate inter-parameter statistics in 10_000-step-intervals
            if step_n_next < 10_000:
                continue

            cache_dir_last = f"models/pythia-{model_size}/step{step_n_next - 10_000}"
            model_n = load_model(model_size, step_n_next - 10_000, cache_dir_last)

            all_parameter_values = torch.tensor([])
            all_parameter_values_next = torch.tensor([])
            for (name_n, parameter_n), (name_n_next, parameter_n_next) in zip(
                model_n.named_parameters(), model_n_next.named_parameters()
            ):
                all_parameter_values = torch.cat((all_parameter_values, parameter_n.flatten()))
                all_parameter_values_next = torch.cat((all_parameter_values_next, parameter_n_next.flatten()))

                # Inter-parameter statistics
                results_inter_parameter = add_inter_parameter_statistics(
                    results_inter_parameter, parameter_n, parameter_n_next, name_n, step_n_next - 10_000, step_n_next
                )

            # Add data for all parameters
            results_inter_parameter = add_inter_parameter_statistics(
                results=results_inter_parameter, 
                parameter_now=all_parameter_values_next, 
                parameter_last=all_parameter_values, 
                name="all_parameters", 
                step_last=step - 10_000, 
                step_now=step,
            )

            # Free up storage
            shutil.rmtree(cache_dir_last)
            shutil.rmtree(cache_dir)
            

        # Free up more memory
        shutil.rmtree(cache_dir)
        
        # Save the results
        os.makedirs(f"results/pythia-{model_size}", exist_ok=True)
        df_intra_parameter = pd.DataFrame(results_intra_parameter)
        df_inter_parameter = pd.DataFrame(results_inter_parameter)
        df_histogram = pd.DataFrame(results_histogram)
        df_intra_parameter.to_csv(f"results/pythia-{model_size}/intra_parameter.csv", index=False)
        df_inter_parameter.to_csv(f"results/pythia-{model_size}/inter_parameter.csv", index=False)
        df_histogram.to_csv(f"results/pythia-{model_size}/histogram.csv", index=False)


#######################
# --- LET'S GO!!! --- #
#######################

if __name__ == "__main__":
    main()
