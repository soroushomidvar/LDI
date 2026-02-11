import scipy.stats as stats
import re

def parse_experiment_result(result: str):
    """Extracts mean and standard deviation from the format '0.747±0.080'"""
    match = re.match(r"([0-9\.]+)±([0-9\.]+)", result)
    if match:
        mean, std = map(float, match.groups())
        return mean, std
    else:
        raise ValueError("Invalid format. Expected format: 0.747±0.080")

def t_test_experiments(exp1: str, exp2: str, n: int = 5, alpha: float = 0.05):
    """
    Runs an independent t-test using mean and standard deviation from two experimental settings.
    
    Parameters:
        exp1 (str): First experiment result in 'mean±std' format.
        exp2 (str): Second experiment result in 'mean±std' format.
        n (int): Number of runs per experiment (default: 5).
        alpha (float): Significance level (default: 0.05).
    
    Returns:
        str: A message indicating whether there is a significant difference.
    """
    mean1, std1 = parse_experiment_result(exp1)
    mean2, std2 = parse_experiment_result(exp2)
    
    # Compute t-statistic and p-value
    t_stat = (mean1 - mean2) / ((std1 ** 2 / n) + (std2 ** 2 / n)) ** 0.5
    df = 2 * (n - 1)  # Degrees of freedom for independent t-test with equal sample sizes
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))  # Two-tailed p-value
    
    if p_value < alpha:
        return f"Significant difference found (p = {p_value:.4f})"
    else:
        return f"No significant difference (p = {p_value:.4f})"

# Example usage
exp1 = "0.846±0.111"
exp2 = "0.976±0.015"
print(t_test_experiments(exp1, exp2))
