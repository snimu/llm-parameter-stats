"""Functions for calculating statistics."""

from dataclasses import dataclass

import torch 
from beartype import beartype 
import jaxtyping as jxt
import einsum


@beartype
@dataclass
class ParameterStatistics:
    """Class for storing parameter statistics."""

    mean: jxt.Float[torch.Tensor, "batch"]
    std: jxt.Float[torch.Tensor, "batch"]
    minimum: jxt.Float[torch.Tensor, "batch"]
    maximum: jxt.Float[torch.Tensor, "batch"]

    outliers_percentage: jxt.Float[torch.Tensor, "batch"]
    sparsity_percentage_absolute: jxt.Float[torch.Tensor, "batch"]
    sparsity_percentage_relative: jxt.Float[torch.Tensor, "batch"]


@beartype
def calculate_outliers(
        x: jxt.Float(torch.Tensor, "batch *other_dims"),
        std: jxt.Float[torch.Tensor, "batch"],
        mean: jxt.Float[torch.Tensor, "batch"],
        threshold: jxt.Float = 3.0,
) -> jxt.Float[torch.Tensor, "batch"]:
    batch_size = x.shape[0]
    numels = x.numel() / batch_size

    x_normalized = (x - mean) / (std + torch.finfo(float).eps())
    outliers_percentage = torch.sum(torch.abs(x_normalized) > threshold, dim=1) / numels
    return outliers_percentage


@beartype
def calculate_maximums(
        x: jxt.Float[torch.Tensor, "batch *other_dims"],
) -> jxt.Float[torch.Tensor, "batch"]:
    return einsum.reduce(x, "b ... -> b", "max")


@beartype
def calculate_minimums(
        x: jxt.Float[torch.Tensor, "batch *other_dims"],
) -> jxt.Float[torch.Tensor, "batch"]:
    return einsum.reduce(x, "b ... -> b", "min")


@beartype
def calculate_means(
        x: jxt.Float[torch.Tensor, "batch *other_dims"],
) -> jxt.Float[torch.Tensor, "batch"]:
    return einsum.reduce(x, "b ... -> b", "mean")


@beartype 
def calculate_std(
        x: jxt.Float[torch.Tensor, "batch *other_dims"],
) -> jxt.Float[torch.Tensor, "batch"]:
    reshaped = x.reshape(x.shape[0], -1)
    return torch.std(reshaped, dim=1)


@beartype
def calculate_sparsity(
        x: jxt.Float[torch.Tensor, "batch *other_dims"],
        std: jxt.Float[torch.Tensor, "batch"],
        mean: jxt.Float[torch.Tensor, "batch"],
        threshold_absolute: jxt.Float = 1e-5,
        threshold_relative: jxt.Float = 1e-5,
) -> tuple[jxt.Float[torch.Tensor, "batch"], jxt.Float[torch.Tensor, "batch"]]:
    batch_size = x.shape[0]
    numels = x.numel() / batch_size
    x_normalized = (x - mean) / (std + torch.finfo(float).eps())

    sparsity_percentage_absolute = torch.sum(torch.abs(x) < threshold_absolute, dim=1) / numels
    sparsity_percentage_relative = torch.sum(torch.abs(x_normalized) < threshold_relative, dim=1) / numels
    return sparsity_percentage_absolute, sparsity_percentage_relative


@beartype
def calculate_per_parameter_statistics(model: nn.Module) -> ParameterStatistics:
    parameters = einops.rearrange(list(model.parameters()), "n ... -> n ...")  # to tensor

    means = calculate_means(parameters)
    stds = calculate_std(parameters)
    maximums = calculate_maximums(parameters)
    minimums = calculate_minimums(parameters)

    outliers_percentage = calculate_outliers(parameters, stds, means)
    sparsity_percentage_absolute, sparsity_percentage_relative = calculate_sparsity(parameters, stds, means)

    return ParameterStatistics(
        mean=means,
        std=stds,
        minimum=minimums,
        maximum=maximums,
        outliers_percentage=outliers_percentage,
        sparsity_percentage_absolute=sparsity_percentage_absolute,
        sparsity_percentage_relative=sparsity_percentage_relative,
    )
