"""Analysis of the parameter statistics of the Pythia models."""

import ast
import os 

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from beartype import beartype


# STEPS = [0] + [2**i for i in range(10)] + [i * 1000 for i in range(1, 144)]


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
        except SyntaxError:
            bin_centers = str_to_list(data['bin_centers'].values[0])
            bin_width = data['bin_width'].values[0]
            counts = str_to_list(data['counts'].values[0])
        except IndexError:
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
def analyze_model(model_size: str) -> None:
    df_hist = pd.read_csv(f"results/pythia-{model_size}/histogram.csv")
    df_intra_parameter = pd.read_csv(f"results/pythia-{model_size}/intra_parameter.csv")
    df_inter_parameter = pd.read_csv(f"results/pythia-{model_size}/inter_parameter.csv")

    # Create video of histogram
    histogram_video(df_hist, model_size, "all_parameters")


@beartype 
def analyze_all() -> None:
    model_sizes = [
        "70m", "70m-deduped",
        # "160m", "160m-deduped",
        # "410m", "410m-deduped", 
        # "1b", "1b-deduped",
        # "1.4b", "1.4b-deduped",
        # "2.8b", "2.8b-deduped",
        # "6.9b", "6.9b-deduped",
        # "12b", "12b-deduped",
    ]
    for model_size in model_sizes:
        analyze_model(model_size)


if __name__ == "__main__":
    analyze_all()
