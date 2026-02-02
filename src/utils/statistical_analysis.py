"""Statistical analysis utilities for multi-seed experiments."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute comprehensive descriptive statistics.
    
    Args:
        values: List of values from multiple seeds
        
    Returns:
        Dictionary with mean, std, variance, CI, min, max, median
    """
    if not values or len(values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'variance': np.nan,
            'ci_95_lower': np.nan,
            'ci_95_upper': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'n': 0,
        }
    
    values_array = np.array(values)
    n = len(values_array)
    mean = np.mean(values_array)
    std = np.std(values_array, ddof=1) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    
    # 95% confidence interval
    ci_95_lower = mean - 1.96 * sem
    ci_95_upper = mean + 1.96 * sem
    
    return {
        'mean': float(mean),
        'std': float(std),
        'variance': float(np.var(values_array, ddof=1)) if n > 1 else 0.0,
        'ci_95_lower': float(ci_95_lower),
        'ci_95_upper': float(ci_95_upper),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array)),
        'n': n,
        'sem': float(sem),
    }


def paired_t_test(
    group1: List[float],
    group2: List[float],
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform paired t-test.
    
    Args:
        group1: First group of values (e.g., method A across seeds)
        group2: Second group of values (e.g., method B across seeds)
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dictionary with t-statistic, p-value, degrees of freedom
    """
    if len(group1) != len(group2):
        raise ValueError(f"Groups must have same length: {len(group1)} vs {len(group2)}")
    
    if len(group1) < 2:
        return {'t_statistic': np.nan, 'p_value': np.nan, 'df': 0}
    
    result = stats.ttest_rel(group1, group2, alternative=alternative)
    
    return {
        't_statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'df': len(group1) - 1,
    }


def wilcoxon_test(
    group1: List[float],
    group2: List[float],
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Args:
        group1: First group of values
        group2: Second group of values
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dictionary with statistic and p-value
    """
    if len(group1) != len(group2):
        raise ValueError(f"Groups must have same length: {len(group1)} vs {len(group2)}")
    
    if len(group1) < 3:
        return {'statistic': np.nan, 'p_value': np.nan}
    
    try:
        result = stats.wilcoxon(group1, group2, alternative=alternative)
        return {
            'statistic': float(result.statistic),
            'p_value': float(result.pvalue),
        }
    except ValueError:
        # All differences are zero
        return {'statistic': 0.0, 'p_value': 1.0}


def compute_effect_size(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Cohen's d (positive means group1 > group2)
    """
    if len(group1) < 2 or len(group2) < 2:
        return np.nan
    
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (mean1 - mean2) / pooled_std
    return float(cohens_d)


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[bool], float]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate (default 0.05)
        
    Returns:
        (reject_null, corrected_alpha) where reject_null[i] is True if p_values[i] < corrected_alpha
    """
    n_comparisons = len(p_values)
    corrected_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha
    
    reject_null = [p < corrected_alpha for p in p_values]
    
    return reject_null, corrected_alpha


def shapiro_wilk_test(values: List[float]) -> Dict[str, float]:
    """
    Test normality using Shapiro-Wilk test.
    
    Args:
        values: Sample values
        
    Returns:
        Dictionary with statistic and p-value
    """
    if len(values) < 3:
        return {'statistic': np.nan, 'p_value': np.nan}
    
    result = stats.shapiro(values)
    return {
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
    }


def levene_test(*groups: List[float]) -> Dict[str, float]:
    """
    Test homogeneity of variance using Levene's test.
    
    Args:
        groups: Variable number of groups to compare
        
    Returns:
        Dictionary with statistic and p-value
    """
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return {'statistic': np.nan, 'p_value': np.nan}
    
    result = stats.levene(*groups)
    return {
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
    }


def format_stats_for_table(
    stats: Dict[str, float],
    decimals: int = 2,
    include_ci: bool = False
) -> str:
    """
    Format statistics for LaTeX table.
    
    Args:
        stats: Dictionary from compute_statistics()
        decimals: Number of decimal places
        include_ci: If True, append CI in brackets
        
    Returns:
        LaTeX-formatted string like "48.97 \pm 0.82" or "48.97 \pm 0.82 [48.40, 49.54]"
    """
    mean = stats.get('mean', np.nan)
    std = stats.get('std', np.nan)
    
    if np.isnan(mean):
        return "---"
    
    formatted = f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"
    
    if include_ci:
        ci_lower = stats.get('ci_95_lower', np.nan)
        ci_upper = stats.get('ci_95_upper', np.nan)
        if not np.isnan(ci_lower) and not np.isnan(ci_upper):
            formatted += f" $[{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]$"
    
    return formatted


def get_significance_marker(p_value: float) -> str:
    """
    Get significance marker for p-value.
    
    Args:
        p_value: P-value from statistical test
        
    Returns:
        LaTeX superscript marker: "", "*", "**", or "***"
    """
    if np.isnan(p_value):
        return ""
    elif p_value < 0.001:
        return "^{***}"
    elif p_value < 0.01:
        return "^{**}"
    elif p_value < 0.05:
        return "^{*}"
    else:
        return ""


def interpret_effect_size(cohens_d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        cohens_d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(cohens_d)
    if np.isnan(abs_d):
        return "unknown"
    elif abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def perform_comprehensive_comparison(
    method_results: Dict[str, List[float]],
    baseline_key: str = 'full_ft',
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Perform comprehensive statistical comparison of all methods against baseline.
    
    Args:
        method_results: Dict mapping method name to list of values across seeds
        baseline_key: Key for baseline method
        alpha: Significance level
        
    Returns:
        Dictionary with statistics and test results for each method
    """
    if baseline_key not in method_results:
        raise ValueError(f"Baseline '{baseline_key}' not found in method_results")
    
    baseline_values = method_results[baseline_key]
    results = {}
    
    # Collect p-values for Bonferroni correction
    comparison_methods = [k for k in method_results.keys() if k != baseline_key]
    p_values_for_correction = []
    
    for method_key in method_results.keys():
        method_values = method_results[method_key]
        
        # Descriptive statistics
        stats_dict = compute_statistics(method_values)
        
        # Comparison with baseline
        if method_key != baseline_key:
            t_test_result = paired_t_test(method_values, baseline_values)
            wilcoxon_result = wilcoxon_test(method_values, baseline_values)
            effect_size = compute_effect_size(method_values, baseline_values)
            
            stats_dict['t_test'] = t_test_result
            stats_dict['wilcoxon'] = wilcoxon_result
            stats_dict['cohens_d'] = effect_size
            stats_dict['effect_size_interpretation'] = interpret_effect_size(effect_size)
            
            p_values_for_correction.append(t_test_result['p_value'])
        else:
            stats_dict['t_test'] = None
            stats_dict['wilcoxon'] = None
            stats_dict['cohens_d'] = None
            stats_dict['effect_size_interpretation'] = None
        
        results[method_key] = stats_dict
    
    # Apply Bonferroni correction
    if p_values_for_correction:
        reject_null, corrected_alpha = bonferroni_correction(p_values_for_correction, alpha)
        
        idx = 0
        for method_key in comparison_methods:
            if method_key in results and results[method_key]['t_test']:
                results[method_key]['bonferroni_reject'] = reject_null[idx]
                results[method_key]['bonferroni_alpha'] = corrected_alpha
                idx += 1
    
    return results


def check_assumptions(
    method_results: Dict[str, List[float]]
) -> Dict[str, Dict[str, Any]]:
    """
    Check statistical assumptions (normality, homogeneity of variance).
    
    Args:
        method_results: Dict mapping method name to list of values
        
    Returns:
        Dictionary with test results for each assumption
    """
    assumptions = {}
    
    # Test normality for each method
    for method_key, values in method_results.items():
        shapiro_result = shapiro_wilk_test(values)
        assumptions[method_key] = {
            'normality': shapiro_result,
            'is_normal': shapiro_result['p_value'] > 0.05 if not np.isnan(shapiro_result['p_value']) else None,
        }
    
    # Test homogeneity of variance across all methods
    all_groups = list(method_results.values())
    levene_result = levene_test(*all_groups)
    assumptions['homogeneity_of_variance'] = {
        'levene': levene_result,
        'is_homogeneous': levene_result['p_value'] > 0.05 if not np.isnan(levene_result['p_value']) else None,
    }
    
    return assumptions
