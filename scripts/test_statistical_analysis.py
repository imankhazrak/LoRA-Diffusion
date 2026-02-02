#!/usr/bin/env python3
"""Test statistical analysis functions with synthetic data."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.statistical_analysis import (
    compute_statistics,
    paired_t_test,
    wilcoxon_test,
    compute_effect_size,
    bonferroni_correction,
    shapiro_wilk_test,
    levene_test,
    format_stats_for_table,
    get_significance_marker,
    interpret_effect_size,
    perform_comprehensive_comparison,
    check_assumptions,
)


def test_basic_statistics():
    """Test basic statistical computations."""
    print("Testing basic statistics...")
    
    # Synthetic data: 10 seeds
    values = [48.5, 49.2, 48.8, 49.1, 48.7, 49.0, 48.9, 49.3, 48.6, 49.4]
    
    stats = compute_statistics(values)
    
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std: {stats['std']:.4f}")
    print(f"  Variance: {stats['variance']:.4f}")
    print(f"  95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]")
    print(f"  Min/Max: {stats['min']:.4f} / {stats['max']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  n: {stats['n']}")
    
    assert stats['n'] == 10
    assert 48.0 < stats['mean'] < 50.0
    assert stats['std'] > 0
    
    print("✓ Basic statistics test passed\n")


def test_significance_testing():
    """Test significance testing."""
    print("Testing significance tests...")
    
    # Group 1: baseline (higher performance)
    group1 = [51.0, 51.5, 50.8, 51.2, 51.1, 51.3, 50.9, 51.4, 51.0, 51.2]
    
    # Group 2: method (lower performance)
    group2 = [48.5, 49.2, 48.8, 49.1, 48.7, 49.0, 48.9, 49.3, 48.6, 49.4]
    
    # Paired t-test
    t_result = paired_t_test(group1, group2)
    print(f"  T-test: t={t_result['t_statistic']:.4f}, p={t_result['p_value']:.4f}, df={t_result['df']}")
    
    # Wilcoxon test
    w_result = wilcoxon_test(group1, group2)
    print(f"  Wilcoxon: statistic={w_result['statistic']:.4f}, p={w_result['p_value']:.4f}")
    
    # Effect size
    cohens_d = compute_effect_size(group1, group2)
    interpretation = interpret_effect_size(cohens_d)
    print(f"  Cohen's d: {cohens_d:.4f} ({interpretation})")
    
    # Significance marker
    sig_marker = get_significance_marker(t_result['p_value'])
    print(f"  Significance marker: {sig_marker}")
    
    assert t_result['p_value'] < 0.05, "Should be significant"
    assert abs(cohens_d) > 0.5, "Should have medium or large effect"
    
    print("✓ Significance testing test passed\n")


def test_bonferroni_correction():
    """Test Bonferroni correction."""
    print("Testing Bonferroni correction...")
    
    p_values = [0.02, 0.03, 0.04, 0.15]
    reject, corrected_alpha = bonferroni_correction(p_values, alpha=0.05)
    
    print(f"  Original alpha: 0.05")
    print(f"  Corrected alpha: {corrected_alpha:.4f}")
    print(f"  P-values: {p_values}")
    print(f"  Reject null: {reject}")
    
    assert corrected_alpha == 0.0125
    assert reject[0] == False  # 0.02 > 0.0125
    assert reject[3] == False  # 0.15 > 0.0125
    
    print("✓ Bonferroni correction test passed\n")


def test_assumption_checks():
    """Test assumption checking."""
    print("Testing assumption checks...")
    
    # Normal-ish data
    np.random.seed(42)
    values = np.random.normal(50, 2, 10).tolist()
    
    shapiro_result = shapiro_wilk_test(values)
    print(f"  Shapiro-Wilk: statistic={shapiro_result['statistic']:.4f}, p={shapiro_result['p_value']:.4f}")
    
    # Multiple groups
    group1 = np.random.normal(50, 2, 10).tolist()
    group2 = np.random.normal(48, 2, 10).tolist()
    group3 = np.random.normal(49, 2, 10).tolist()
    
    levene_result = levene_test(group1, group2, group3)
    print(f"  Levene: statistic={levene_result['statistic']:.4f}, p={levene_result['p_value']:.4f}")
    
    print("✓ Assumption checks test passed\n")


def test_comprehensive_comparison():
    """Test comprehensive comparison."""
    print("Testing comprehensive comparison...")
    
    np.random.seed(42)
    
    method_results = {
        'full_ft': np.random.normal(51, 0.7, 10).tolist(),
        'lora_diffusion': np.random.normal(49, 0.8, 10).tolist(),
        'weight_lora': np.random.normal(48.5, 1.0, 10).tolist(),
        'adapters': np.random.normal(49.8, 0.9, 10).tolist(),
        'bitfit': np.random.normal(48, 1.2, 10).tolist(),
    }
    
    results = perform_comprehensive_comparison(method_results, baseline_key='full_ft')
    
    print(f"  Methods analyzed: {len(results)}")
    
    for method, stats in results.items():
        mean = stats['mean']
        std = stats['std']
        print(f"  {method:15s}: {mean:.2f} ± {std:.2f}", end="")
        
        if stats['t_test']:
            p_val = stats['t_test']['p_value']
            cohens_d = stats['cohens_d']
            print(f"  (p={p_val:.4f}, d={cohens_d:.3f})")
        else:
            print("  (baseline)")
    
    assert 'full_ft' in results
    assert results['lora_diffusion']['t_test'] is not None
    
    print("✓ Comprehensive comparison test passed\n")


def test_latex_formatting():
    """Test LaTeX formatting."""
    print("Testing LaTeX formatting...")
    
    stats = {
        'mean': 48.97,
        'std': 0.82,
        'ci_95_lower': 48.40,
        'ci_95_upper': 49.54,
    }
    
    formatted = format_stats_for_table(stats, decimals=2, include_ci=False)
    print(f"  Without CI: {formatted}")
    
    formatted_ci = format_stats_for_table(stats, decimals=2, include_ci=True)
    print(f"  With CI: {formatted_ci}")
    
    assert "48.97" in formatted
    assert "0.82" in formatted
    assert "48.40" in formatted_ci
    
    print("✓ LaTeX formatting test passed\n")


def main():
    """Run all tests."""
    print("=" * 80)
    print("STATISTICAL ANALYSIS MODULE TESTS")
    print("=" * 80)
    print()
    
    try:
        test_basic_statistics()
        test_significance_testing()
        test_bonferroni_correction()
        test_assumption_checks()
        test_comprehensive_comparison()
        test_latex_formatting()
        
        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print()
        print("The statistical analysis module is working correctly.")
        print("You can now run multi-seed experiments with confidence.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
