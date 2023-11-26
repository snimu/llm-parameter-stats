"""Analysis of the parameter statistics of the Pythia models."""

import sys
import ast
import os 
from collections.abc import Sequence

import torch
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
try:
    from beartype import beartype
except ImportError:
    pass
from packaging import version


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
def str_to_list(text: str) -> list[float]:
    """Convert a string to a list of floats."""
    text = text.strip()
    text = text.replace(" ", ",")
    text = ",".join(text.split("\n"))
    while ",," in text:
        text = text.replace(",,", ",")
    text = text.replace("[,", "[")
    text = text.replace(",]", "]")

    return ast.literal_eval(text)


@save_beartype
def histogram_video(df: pd.DataFrame, model_size: str, parameter: str) -> None:
    """
    Create a video of the histograms.
    
    Parameters
    ----------

    df : pd.DataFrame
        The dataframe containing the histograms.
        Created from the following dict:
            results_histogram = {
                "parameter": [],
                "step": [],
                "counts": [],
                "bin_centers": [],
                "bin_width": [],
            }
        'parameter' is the name of the parameter.
        'step' is the training step of the model.
        'counts' is the number of parameters in each bin.
        'bin_centers' is the center of each bin.
        'bin_width' is the width of each bin.
    model_size : str
        The size of the model. Used for the filename.
    parameter: str
        The name of the parameter to create the video for.

    """
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Function to update the histogram for each frame
    def update(i):
        ax.clear()
        data = df[(df['step'] == i) & (df['parameter'] == parameter)]

        try:
            bin_centers = ast.literal_eval(data['bin_centers'].values[0])
            bin_width = data['bin_width'].values[0]
            counts = ast.literal_eval(data['counts'].values[0])
        except SyntaxError:  # in case data is saved as np.array
            bin_centers = str_to_list(data['bin_centers'].values[0])
            bin_width = data['bin_width'].values[0]
            counts = str_to_list(data['counts'].values[0])
        except IndexError:  # in case data is saved as np.array
            bin_centers = str_to_list(data['bin_centers'].values[0])
            bin_width = data['bin_width'].values[0]
            counts = str_to_list(data['counts'].values[0])
            
        ax.bar(bin_centers, counts, width=bin_width)
        ax.set_title(f"Step {i}")

    # Create an animation by repeatedly calling a function
    anim = FuncAnimation(fig, update, frames=df['step'].unique(), interval=200)

    # Save the animation as a video file
    os.makedirs(f"results/pythia-{model_size}/histogram", exist_ok=True)
    try:
        anim.save(f"results/pythia-{model_size}/histogram/{parameter}.mp4")
    except ValueError:
        # print("Error: .mp4 format is not supported. Saving the animation as .gif instead.")
        anim.save(f"results/pythia-{model_size}/histogram/{parameter}.gif", writer='pillow')


@save_beartype
def plot_results(
        dfs: Sequence[pd.DataFrame], 
        model_sizes: Sequence[str], 
        df_type: str,
        show: bool = True,
        name_suffix: str = "",
        x_axis: str = "step",
) -> None:
    keys = dfs[0].keys()
    assert df_type in ("intra_parameter", "inter_parameter", "inter_parameter_10_000")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for key in keys:
        if key in ("parameter", "step", "step_next"):
            continue
        for i, (df, model_size) in enumerate(zip(dfs, model_sizes)):
            df = df[df['parameter'] == "all_parameters"]
            plt.plot(
                df[x_axis], 
                df[key], 
                label=model_size, 
                linestyle=":" if "deduped" in model_size else "-",  # deduped models have a dot in the legend
                color=colors[int(i/2)],  # deduped and non-deduped models have the same color
            )
        
        plt.xlabel(x_axis)
        plt.ylabel(key)

        title = f"{key} ({df_type})"
        plt.title(title)
        plt.gcf().subplots_adjust(bottom=0.3)  # Make space for the legend
        plt.legend(
            bbox_to_anchor=(0., -0.3, 1., .102),  # Move the legend down
            loc='upper left',
            ncol=3, 
            mode="expand", 
            borderaxespad=0.
        )

        if show:
            plt.show()
        else:
            os.makedirs(f"results/all/{df_type}", exist_ok=True)  # TODO: think of more consistent naming for dirs
            plt.savefig(f"results/all/{df_type}/{key}{name_suffix}.png", dpi=300)
        plt.cla()
        plt.clf()
        plt.close()


def plot_set_of_results_by_step(
        dfs_intra_parameter: Sequence[pd.DataFrame],
        dfs_inter_parameter: Sequence[pd.DataFrame],
        dfs_inter_parameter_10_000: Sequence[pd.DataFrame] | None,
        model_sizes: Sequence[str],
        show: bool = True,
        name_suffix: str = "",
) -> None:
    plot_results(
        dfs_intra_parameter, 
        model_sizes, 
        df_type="intra_parameter", 
        show=show, 
        x_axis="step",
        name_suffix=name_suffix,
    )
    plot_results(
        dfs_inter_parameter, 
        model_sizes, 
        df_type="inter_parameter", 
        show=show, 
        x_axis="step",
        name_suffix=name_suffix,
    )

    if dfs_inter_parameter_10_000 is None:
        return 

    plot_results(
        dfs_inter_parameter_10_000, 
        model_sizes, 
        df_type="inter_parameter_10_000", 
        show=show, 
        x_axis="step",
        name_suffix=name_suffix,
    )


def plot_set_of_results_by_chinchillar_perc(
        dfs_intra_parameter: Sequence[pd.DataFrame],
        dfs_inter_parameter: Sequence[pd.DataFrame],
        dfs_inter_parameter_10_000: Sequence[pd.DataFrame] | None,
        model_sizes: Sequence[str],
        show: bool = True,
        name_suffix: str = "",
) -> None:
    plot_results(
        dfs_intra_parameter, 
        model_sizes, 
        df_type="intra_parameter", 
        show=show, 
        x_axis="percentage_of_chinchilla_optimal",
        name_suffix=name_suffix,
    )
    plot_results(
        dfs_inter_parameter, 
        model_sizes, 
        df_type="inter_parameter", 
        show=show, 
        x_axis="percentage_of_chinchilla_optimal",
        name_suffix=name_suffix,
    )

    if dfs_inter_parameter_10_000 is None:
        return 

    plot_results(
        dfs_inter_parameter_10_000, 
        model_sizes, 
        df_type="inter_parameter_10_000", 
        show=show, 
        x_axis="percentage_of_chinchilla_optimal",
        name_suffix=name_suffix,
    )


@save_beartype
def analyze_models(show: bool = True) -> None:
    model_sizes = MODEL_SIZES

    dfs_hist = []
    dfs_intra_parameter = []
    dfs_inter_parameter = []
    dfs_inter_parameter_10_000 = []
    for model_size in model_sizes:
        dfs_hist.append(pd.read_csv(f"results/pythia-{model_size}/histogram.csv"))
        dfs_intra_parameter.append(pd.read_csv(f"results/pythia-{model_size}/intra_parameter.csv"))

        df = pd.read_csv(f"results/pythia-{model_size}/inter_parameter.csv")
        dfs_inter_parameter.append(df[df['step_next'] - df['step'] < 10_000])
        dfs_inter_parameter_10_000.append(df[df['step_next'] - df['step'] == 10_000])

    for i, model_size in enumerate(model_sizes):
        percentage_of_chinchilla_optimal = get_percentage_of_chinchilla_optimal(
            model_size, steps=dfs_intra_parameter[i]['step'].tolist()
        )
        dfs_intra_parameter[i]['percentage_of_chinchilla_optimal'] = percentage_of_chinchilla_optimal
        percentage_of_chinchilla_optimal = get_percentage_of_chinchilla_optimal(
            model_size, steps=dfs_inter_parameter[i]['step_next'].tolist()
        )
        dfs_inter_parameter[i]['percentage_of_chinchilla_optimal'] = percentage_of_chinchilla_optimal
        percentage_of_chinchilla_optimal = get_percentage_of_chinchilla_optimal(
            model_size, steps=dfs_inter_parameter_10_000[i]['step_next'].tolist()
        )
        dfs_inter_parameter_10_000[i]['percentage_of_chinchilla_optimal'] = percentage_of_chinchilla_optimal

    # Create video of histogram
    # for i, model_size in enumerate(model_sizes):
    #     histogram_video(dfs_hist[i], model_size, "all_parameters")

    # Plot results
    plot_set_of_results_by_step(
        dfs_intra_parameter, 
        dfs_inter_parameter, 
        dfs_inter_parameter_10_000, 
        model_sizes, 
        show=show,
        name_suffix="_by_step",
    )
    plot_set_of_results_by_chinchillar_perc(
        dfs_intra_parameter, 
        dfs_inter_parameter, 
        dfs_inter_parameter_10_000, 
        model_sizes, 
        show=show,
        name_suffix="_by_chinchilla_perc",
    )

    # Plot results for the first 1000 steps
    # First, reduce the dfs s.t. they only contain the first 1000 steps
    dfs_intra_parameter_first_1000 = [df[df['step'] <= 1000] for df in dfs_intra_parameter]
    dfs_inter_parameter_first_1000 = [df[df['step'] <= 1000] for df in dfs_inter_parameter]

    # Then, plot the results
    plot_set_of_results_by_step(
        dfs_inter_parameter=dfs_inter_parameter_first_1000,
        dfs_intra_parameter=dfs_intra_parameter_first_1000,
        dfs_inter_parameter_10_000=None,
        model_sizes=model_sizes,
        show=show,
        name_suffix="_1000_steps_by_step",
    )

    # Plot the results for the first 100% (so 1.0) of chinchillar optimal steps
    dfs_intra_parameter_first_100_perc = [
        df[df['percentage_of_chinchilla_optimal'] <= 1.0]
        for df in dfs_intra_parameter
    ]
    dfs_inter_parameter_first_100_perc = [
        df[df['percentage_of_chinchilla_optimal'] <= 1.0] 
        for df in dfs_inter_parameter
    ]

    plot_set_of_results_by_chinchillar_perc(
        dfs_inter_parameter=dfs_inter_parameter_first_100_perc,
        dfs_intra_parameter=dfs_intra_parameter_first_100_perc,
        dfs_inter_parameter_10_000=None,
        model_sizes=model_sizes,
        show=show,
        name_suffix="_100_perc_by_chinchilla_perc",
    )


if __name__ == "__main__":
    analyze_models(show=False)

