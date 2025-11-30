"""
Statistical Analysis Module

Provides statistical testing and analysis functions for research experiments.

Functions:
    anova_test: One-way ANOVA
    t_test: Independent and paired t-tests
    correlation: Pearson and Spearman correlation
    regression: Linear and nonlinear regression
    effect_size: Cohen's d and eta-squared
    confidence_interval: Calculate confidence intervals
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from scipy import stats
from loguru import logger


def anova_test(
    groups: List[np.ndarray], 
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform one-way ANOVA test.
    
    Args:
        groups: List of data arrays (one per group)
        alpha: Significance level
        
    Returns:
        Dictionary with F-statistic, p-value, and effect size
    """
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate eta-squared (effect size)
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum(np.sum((g - grand_mean)**2) for g in groups)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    result = {
        "f_statistic": f_stat,
        "p_value": p_value,
        "eta_squared": eta_squared,
        "significant": p_value < alpha,
        "num_groups": len(groups)
    }
    
    logger.info(f"ANOVA: F={f_stat:.4f}, p={p_value:.4f}, η²={eta_squared:.4f}")
    return result


def t_test(
    group1: np.ndarray, 
    group2: np.ndarray, 
    paired: bool = False,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform t-test (independent or paired).
    
    Args:
        group1: First data array
        group2: Second data array
        paired: Whether to use paired t-test
        alpha: Significance level
        
    Returns:
        Dictionary with t-statistic, p-value, and Cohen's d
    """
    if paired:
        t_stat, p_value = stats.ttest_rel(group1, group2)
    else:
        t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Calculate Cohen's d
    cohens_d = cohen_d(group1, group2)
    
    result = {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < alpha,
        "paired": paired
    }
    
    logger.info(f"t-test: t={t_stat:.4f}, p={p_value:.4f}, d={cohens_d:.4f}")
    return result


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        group1: First data array
        group2: Second data array
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def correlation(
    x: np.ndarray, 
    y: np.ndarray, 
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Calculate correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
        method: 'pearson' or 'spearman'
        
    Returns:
        Dictionary with correlation coefficient and p-value
    """
    if method == "pearson":
        r, p_value = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p_value = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    result = {
        "correlation": r,
        "p_value": p_value,
        "method": method
    }
    
    logger.info(f"{method.capitalize()} correlation: r={r:.4f}, p={p_value:.4f}")
    return result


def confidence_interval(
    data: np.ndarray, 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean.
    
    Args:
        data: Data array
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return (mean - margin, mean + margin)


def regression_linear(
    x: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Perform linear regression.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        Dictionary with regression results
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    result = {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err
    }

    logger.info(f"Linear regression: R²={result['r_squared']:.4f}, p={p_value:.4f}")
    return result


class StatisticalAnalyzer:
    """
    Statistical analyzer class for experiments.

    Provides methods for statistical testing and analysis with consistent interface.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.

        Args:
            alpha: Significance level for tests
        """
        self.alpha = alpha
        logger.info(f"StatisticalAnalyzer initialized with alpha={alpha}")

    def t_test(
        self,
        group1: List[float],
        group2: List[float],
        paired: bool = False,
        alternative: str = "two-sided"
    ) -> Dict[str, Any]:
        """
        Perform t-test.

        Args:
            group1: First group of values
            group2: Second group of values
            paired: Whether to use paired t-test
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with test results
        """
        arr1 = np.array(group1)
        arr2 = np.array(group2)

        if paired:
            t_stat, p_value = stats.ttest_rel(arr1, arr2, alternative=alternative)
        else:
            t_stat, p_value = stats.ttest_ind(arr1, arr2, alternative=alternative)

        # Calculate Cohen's d
        cohens_d_value = cohen_d(arr1, arr2)

        result = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d_value,
            "significant": p_value < self.alpha,
            "paired": paired,
            "alternative": alternative
        }

        logger.info(f"t-test: t={t_stat:.4f}, p={p_value:.4f}, d={cohens_d_value:.4f}")
        return result

    def anova(
        self,
        groups: List[List[float]],
        group_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform one-way ANOVA.

        Args:
            groups: List of groups (each group is a list of values)
            group_names: Optional names for groups

        Returns:
            Dictionary with ANOVA results
        """
        arrays = [np.array(g) for g in groups]
        f_stat, p_value = stats.f_oneway(*arrays)

        # Calculate eta-squared (effect size)
        grand_mean = np.mean(np.concatenate(arrays))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in arrays)
        ss_total = sum(np.sum((g - grand_mean)**2) for g in arrays)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        result = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "eta_squared": eta_squared,
            "significant": p_value < self.alpha,
            "num_groups": len(groups),
            "group_names": group_names
        }

        logger.info(f"ANOVA: F={f_stat:.4f}, p={p_value:.4f}, η²={eta_squared:.4f}")
        return result

    def correlation(
        self,
        x: List[float],
        y: List[float],
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        Calculate correlation.

        Args:
            x: First variable
            y: Second variable
            method: 'pearson' or 'spearman'

        Returns:
            Dictionary with correlation results
        """
        arr_x = np.array(x)
        arr_y = np.array(y)

        return correlation(arr_x, arr_y, method=method)

    def regression(
        self,
        x: List[float],
        y: List[float]
    ) -> Dict[str, Any]:
        """
        Perform linear regression.

        Args:
            x: Independent variable
            y: Dependent variable

        Returns:
            Dictionary with regression results
        """
        arr_x = np.array(x)
        arr_y = np.array(y)

        return regression_linear(arr_x, arr_y)
