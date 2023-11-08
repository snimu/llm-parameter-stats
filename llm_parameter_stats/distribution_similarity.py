import torch 
from scipy import stats
try:
    from beartype import beartype_this_package
    beartype_this_package()
except ImportError:
    pass


def distribution_similarity_ks(tensor: torch.Tensor, distribution: str = "norm") -> tuple[float, float]:
    assert distribution in ["norm", "expon", "logistic", "gumbel_l", "gumbel_r"]
    data = to_numpy(tensor.flatten())
    # Calculate the parameters for the KS test based on the distribution
    if distribution == "expon":
        # For exponential, scale is the mean of the data, location is 0 by default
        args = (0, data.mean())
    elif distribution in ["gumbel_l", "gumbel_r"]:
        # For Gumbel distributions, estimate the parameters using MLE
        args = stats.gumbel_r.fit(data) if distribution == "gumbel_r" else stats.gumbel_l.fit(data)
    else:
        # For other distributions, use the sample mean and standard deviation
        args = (data.mean(), data.std())
    result = stats.kstest(data, distribution, args=args)
    return result.statistic, result.pvalue


def distribution_similarity_anderson(tensor: torch.Tensor, distribution: str = "norm") -> tuple[float, str]:
    assert distribution in ["norm", "expon", "logistic", "gumbel_l", "gumbel_r"]
    data = to_numpy(tensor.flatten())
    result = stats.anderson(data, dist=distribution)
    significance_level = None
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            significance_level = sl
            break
    return result.statistic, significance_level

 