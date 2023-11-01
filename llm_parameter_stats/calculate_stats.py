import os
import shutil
import itertools
from tqdm import tqdm

import rich
from beartype import beartype
import torch
from torch import nn
import pandas as pd
from transformers import GPTNeoXForCausalLM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_HISTOGRAM_BINS = 100


@beartype 
def add_parameter_statistics(
        results: dict[str, torch.Tensor],
        parameter: nn.Parameter | torch.Tensor,
        name: str, 
        step: int,
        step_next: int | None = None,
) -> dict[str, torch.Tensor]:
    results["parameter"].append(name)
    results["step"].append(step)
    if step_next is not None:
        results["step_next"].append(step_next)

    prefix = "" if step_next is None else "cos_sim_"
    results[f"{prefix}mean"].append(torch.mean(parameter).detach().cpu().numpy())
    results[f"{prefix}std"].append(torch.std(parameter).detach().cpu().numpy())
    results[f"{prefix}maximum"].append(torch.max(parameter).detach().cpu().numpy())
    results[f"{prefix}minimum"].append(torch.min(parameter).detach().cpu().numpy())
    results[f"{prefix}abs_mean"].append(torch.mean(torch.abs(parameter)).detach().cpu().numpy())
    results[f"{prefix}abs_std"].append(torch.std(torch.abs(parameter)).detach().cpu().numpy())
    results[f"{prefix}abs_maximum"].append(torch.max(torch.abs(parameter)).detach().cpu().numpy())
    results[f"{prefix}abs_minimum"].append(torch.min(torch.abs(parameter)).detach().cpu().numpy())
    return results


@beartype
def add_histogram(
        results: dict[str, torch.Tensor],
        parameter: nn.Parameter | torch.Tensor,
        name: str,
        step: int,
) -> dict[str, torch.Tensor]:
    counts, bin_edges = torch.histogram(parameter, bins=NUM_HISTOGRAM_BINS)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    results["parameter"].append(name)
    results["step"].append(step)
    results["counts"].append(list(counts.detach().cpu().numpy()))
    results["bin_centers"].append(list(bin_centers.detach().cpu().numpy()))
    results["bin_width"].append(bin_width.item())

    return results


@beartype
def main() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # model_sizes = [
    #     "70m", "70m-deduped",
    #     "160m", "160m-deduped",
    #     "410m", "410m-deduped", 
    #     "1b", "1b-deduped",
    #     "1.4b", "1.4b-deduped",
    #     "2.8b", "2.8b-deduped",
    #     "6.9b", "6.9b-deduped",
    #     "12b", "12b-deduped",
    # ]
    steps = [0] + [2**i for i in range(10)] + [i * 1000 for i in range(1, 144)]
    model_sizes = ["70m", "70m-deduped"]

    for model_size in model_sizes:
        title = "| ANALYZING NEW MODEL SIZE |"
        width = len(title)
        title = f"\n\n{'=' * width}\n{title}\n{'=' * width}\n"
        title += f"| Size: {model_size} |\n\n"

        print(title)

        results_intra_parameter = {
            "parameter": [],
            "step": [],
            "mean": [],
            "std": [],
            "maximum": [],
            "minimum": [],
            "abs_mean": [],
            "abs_std": [],
            "abs_maximum": [],
            "abs_minimum": [],
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
            "cos_sim_mean": [],
            "cos_sim_std": [],
            "cos_sim_maximum": [],
            "cos_sim_minimum": [],
            "cos_sim_abs_mean": [],
            "cos_sim_abs_std": [],
            "cos_sim_abs_maximum": [],
            "cos_sim_abs_minimum": [],
        }

        for i, (step_n, step_n_next) in enumerate(itertools.pairwise(steps)):
            rich.print(f"\nStep: {step_n=}, {step_n_next=} :: number {i+1}/{len(steps)}\n")

            cache_dir_last = f"models/pythia-{model_size}/step{step_n}"
            cache_dir = f"models/pythia-{model_size}/step{step_n_next}"

            # Load the models
            if step_n == 0:
                model_n = GPTNeoXForCausalLM.from_pretrained(
                    f"EleutherAI/pythia-{model_size}",
                    revision=f"step{step_n}",
                    cache_dir=cache_dir_last,
                )
            else:
                model_n = model_n_next

            model_n_next = GPTNeoXForCausalLM.from_pretrained(
                f"EleutherAI/pythia-{model_size}",
                revision=f"step{step_n_next}",
                cache_dir=cache_dir,
            )

            all_parameter_values = torch.tensor([])
            all_parameter_values_next = torch.tensor([])

            # Calculate the statistics
            for (name_n, parameter_n), (name_n_next, parameter_n_next) in zip(
                model_n.named_parameters(), model_n_next.named_parameters()
            ):
                # Speed up by moving to GPU if available
                parameter_n = parameter_n.to(DEVICE)
                parameter_n_next = parameter_n_next.to(DEVICE)

                # Intra-parameter statistics
                results_intra_parameter = add_parameter_statistics(results_intra_parameter, parameter_n, name_n, step_n)
                all_parameter_values = torch.cat((all_parameter_values, parameter_n.flatten()))
                results_histogram = add_histogram(results_histogram, parameter_n, name_n, step_n)
                if step_n_next == steps[-1]:
                    results_intra_parameter = add_parameter_statistics(results_intra_parameter, parameter_n_next, name_n_next, step_n_next)
                    results_histogram = add_histogram(results_histogram, parameter_n_next, name_n_next, step_n_next)
                    # The histogram of the values has to happen independently for every step, 
                    #   so we have to save the values for the next step independently as well.
                    all_parameter_values_next = torch.cat((all_parameter_values_next, parameter_n_next.flatten()))

                # Inter-parameter statistics
                cos_sim = torch.nn.functional.cosine_similarity(
                    parameter_n.flatten(), parameter_n_next.flatten(), dim=0
                )

                results_inter_parameter = add_parameter_statistics(results_inter_parameter, cos_sim, name_n, step_n, step_n_next)

                # Free up GPU memory
                parameter_n = parameter_n.cpu()
                parameter_n_next = parameter_n_next.cpu()

            # Add data for all parameters
            results_intra_parameter = add_parameter_statistics(results_intra_parameter, all_parameter_values, "all_parameters", step_n)
            results_histogram = add_histogram(results_histogram, all_parameter_values, "all_parameters", step_n)
            if step_n_next == steps[-1]:
                results_intra_parameter = add_parameter_statistics(results_intra_parameter, all_parameter_values_next, "all_parameters", step_n_next)
                results_histogram = add_histogram(results_histogram, all_parameter_values_next, "all_parameters", step_n_next)
            else:
                results_inter_parameter = add_parameter_statistics(results_inter_parameter, all_parameter_values, "all_parameters", step_n, step_n_next)

            # Free up storage
            shutil.rmtree(cache_dir_last)

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


if __name__ == "__main__":
    main()
