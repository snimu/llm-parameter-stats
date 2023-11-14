"""Analysis of the parameter statistics of the Pythia models."""

import ast
import os 
from collections.abc import Sequence

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from beartype import beartype


# STEPS = [0] + [2**i for i in range(10)] + [i * 1000 for i in range(1, 144)]
PYTHIA_BATCH_SIZE = 2e6  # 2 million


@beartype
def get_chinchilla_optimal_steps(
        model_size: str,
) -> int:
    model_size = model_size.split("pythia-")[1]
    if "m" in model_size: 
        factor = 1e6
    elif "b" in model_size:
        factor = 1e9
    else:
        raise ValueError(f"Model size {model_size} not supported.")
    model_size = int(model_size.split("m")[0]) * factor

    num_samples_chinchilla_optimal = int(20 * model_size)
    num_steps_chinchilla_optimal = int(num_samples_chinchilla_optimal / PYTHIA_BATCH_SIZE)
    return num_steps_chinchilla_optimal


@beartype
def get_percentage_of_chinchilla_optimal(
        model_size: str,
        steps: Sequence[int],
) -> np.ndarray:
    num_steps_chinchilla_optimal = get_chinchilla_optimal_steps(model_size)
    return np.array(steps) / num_steps_chinchilla_optimal


@beartype
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


@beartype
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


@beartype
def plot_results(
        dfs: Sequence[pd.DataFrame], 
        model_sizes: Sequence[str], 
        df_type: str,
        show: bool = True,
) -> None:
    keys = dfs[0].keys()
    assert df_type in ("intra_parameter", "inter_parameter", "inter_parameter_10_000")

    for key in keys:
        if key in ("parameter", "step", "step_next"):
            continue
        for df, model_size in zip(dfs, model_sizes):
            df = df[df['parameter'] == "all_parameters"]
            plt.plot(df['step'], df[key], label=model_size)
        
        plt.xlabel("Step")
        plt.ylabel(key)

        title = f"{key} ({df_type})"
        plt.title(title)
        plt.legend()

        if show:
            plt.show()
        else:
            os.makedirs(f"results/all/{df_type}", exist_ok=True)  # TODO: think of more consistent naming for dirs
            plt.savefig(f"results/all/{df_type}/{key}.png", dpi=300)
        plt.cla()
        plt.clf()
        plt.close()


@beartype
def analyze_models(show: bool = True) -> None:
    model_sizes = [
        "70m", "70m-deduped",
        "160m", "160m-deduped",
        "410m", "410m-deduped", 
        "1b", "1b-deduped",
        # "1.4b", "1.4b-deduped",
        # "2.8b", "2.8b-deduped",
        # "6.9b", "6.9b-deduped",
        # "12b", "12b-deduped",
    ]

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

    # Create video of histogram
    for i, model_size in enumerate(model_sizes):
        histogram_video(dfs_hist[i], model_size, "all_parameters")

    # Plot results
    plot_results(dfs_intra_parameter, model_sizes, df_type="intra_parameter", show=show)
    plot_results(dfs_inter_parameter, model_sizes, df_type="inter_parameter", show=show)
    plot_results(dfs_inter_parameter_10_000, model_sizes, df_type="inter_parameter_10_000", show=show)



if __name__ == "__main__":
    analyze_models(show=False)

