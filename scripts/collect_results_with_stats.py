#!/usr/bin/env python3
"""Collect multi-seed experiment results and compute statistics."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np

# Load statistical_analysis without importing src.utils (avoids torch dependency)
_script_dir = Path(__file__).resolve().parent
_sa_path = _script_dir.parent / "src" / "utils" / "statistical_analysis.py"
_spec = importlib.util.spec_from_file_location("statistical_analysis", _sa_path)
_sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sa)
compute_statistics = _sa.compute_statistics
perform_comprehensive_comparison = _sa.perform_comprehensive_comparison
check_assumptions = _sa.check_assumptions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect multi-seed results and compute statistics"
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing experiment outputs",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sst2",
        help="Task name",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit"],
        help="List of methods to collect",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42-51",
        help="Seed range (e.g., '42-51') or list (e.g., '42 43 44')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="collected_results_with_stats.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="full_ft",
        help="Baseline method for significance testing",
    )
    
    return parser.parse_args()


def parse_seed_spec(seed_spec: str) -> List[int]:
    """
    Parse seed specification.
    
    Args:
        seed_spec: Either "42-51" (range) or "42 43 44" (list)
        
    Returns:
        List of seed integers
    """
    if '-' in seed_spec and ' ' not in seed_spec:
        # Range format: "42-51"
        start, end = map(int, seed_spec.split('-'))
        return list(range(start, end + 1))
    else:
        # List format: "42 43 44"
        return [int(s) for s in seed_spec.split()]


def find_experiment_output(
    base_dir: Path,
    task: str,
    method: str,
    seed: int
) -> Optional[Path]:
    """Find output directory for a specific experiment."""
    # Try exact match first
    exact_match = base_dir / f"{task}_{method}_seed{seed}"
    if exact_match.exists():
        return exact_match
    
    # Try pattern matching
    pattern = f"{task}_{method}_seed{seed}*"
    matches = list(base_dir.glob(pattern))
    if matches:
        return matches[0]
    
    return None


# Tasks where we report generation-based (task) accuracy as val_accuracy when available.
# Token-level denoising accuracy can be ~100% even when task accuracy is lower; for GLUE-style reporting we use generation.
CLASSIFICATION_TASKS = frozenset({"sst2", "qnli", "agnews", "mrpc", "cola", "rte", "mnli"})

def extract_metrics_from_output(output_dir: Path, task: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Extract metrics from experiment output directory.
    For classification tasks, val_accuracy uses generation_accuracy when present (standard task metric)."""
    metrics = {}
    
    # Training summary
    summary_file = output_dir / "training_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
            final_train = summary.get("final_training_metrics", {})
            metrics["train_loss"] = final_train.get("loss")
            metrics["train_accuracy"] = final_train.get("accuracy")
            metrics["total_steps"] = summary.get("total_steps")
    
    # Training history
    history_file = output_dir / "training_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
            if history:
                initial = history[0].get("metrics", {})
                final = history[-1].get("metrics", {})
                metrics["initial_loss"] = initial.get("loss")
                metrics["final_loss"] = final.get("loss")
                metrics["final_train_acc"] = final.get("accuracy")
    
    # Evaluation results: for classification tasks use generation_accuracy as val_accuracy when present (GLUE-style task metric).
    eval_file = output_dir / "eval_results.json"
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
            eval_metrics = eval_data.get("metrics", {})
            gen_acc = eval_metrics.get("generation_accuracy")
            token_acc = eval_metrics.get("accuracy")
            if task and task in CLASSIFICATION_TASKS and gen_acc is not None:
                metrics["val_accuracy"] = gen_acc  # generation-based (task) accuracy
                metrics["val_token_accuracy"] = token_acc  # token-level denoising (for reference)
            else:
                metrics["val_accuracy"] = token_acc  # token-level (default / non-classification)
            metrics["val_generation_accuracy"] = gen_acc  # for reference
            metrics["val_f1"] = eval_metrics.get("f1")
            metrics["val_loss"] = eval_metrics.get("loss")
    
    # Evaluation history (fallback)
    eval_history_file = output_dir / "evaluation_history.json"
    if eval_history_file.exists() and metrics.get("val_accuracy") is None:
        with open(eval_history_file, "r") as f:
            eval_history = json.load(f)
            if eval_history:
                # Get best validation accuracy
                best_eval = max(
                    eval_history,
                    key=lambda x: x.get("metrics", {}).get("accuracy", 0)
                )
                token_acc = best_eval.get("metrics", {}).get("accuracy", 0)
                if token_acc > 0.1:  # Task-level accuracy
                    metrics["val_accuracy"] = token_acc
    
    # Classification head results (optional; evaluate.py writes .classification_head.json when --eval_classification_head)
    for ch_name in ("eval_results.classification_head.json", "eval_results_ch.classification_head.json"):
        ch_file = output_dir / ch_name
        if ch_file.exists():
            with open(ch_file, "r") as f:
                ch_data = json.load(f)
                metrics["classification_head_val_acc"] = ch_data.get("classification_head_val_acc")
                metrics["classification_head_test_acc"] = ch_data.get("classification_head_test_acc")
            break

    return metrics if metrics else None


def collect_multi_seed_results(
    base_dir: Path,
    task: str,
    methods: List[str],
    seeds: List[int]
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, List[int]]]:
    """
    Collect results from all seed runs.
    
    Returns:
        (results, method_seeds): results is method -> metric_name -> list of values;
        method_seeds[method] is list of seeds in same order as values (for alignment).
    """
    results = defaultdict(lambda: defaultdict(list))
    method_seeds: Dict[str, List[int]] = defaultdict(list)
    missing_runs = []

    for method in methods:
        for seed in seeds:
            output_dir = find_experiment_output(base_dir, task, method, seed)

            if output_dir is None:
                missing_runs.append((method, seed))
                print(f"Warning: Missing output for {task}_{method}_seed{seed}")
                continue

            metrics = extract_metrics_from_output(output_dir, task=task)

            if metrics is None:
                missing_runs.append((method, seed))
                print(f"Warning: No metrics found for {task}_{method}_seed{seed}")
                continue

            method_seeds[method].append(seed)
            for metric_name, value in metrics.items():
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    results[method][metric_name].append(value)

    if missing_runs:
        print(f"\nTotal missing runs: {len(missing_runs)}")
        for method, seed in missing_runs:
            print(f"  - {method}_seed{seed}")

    return dict(results), dict(method_seeds)


def _align_by_common_seeds(
    method_values: Dict[str, List[float]],
    method_seeds: Dict[str, List[int]]
) -> Dict[str, List[float]]:
    """Align method values by common seeds so paired comparisons are valid."""
    if not method_values or not method_seeds:
        return method_values
    common = set(method_seeds.get(next(iter(method_values)), []))
    for m in method_values:
        common &= set(method_seeds.get(m, []))
    common_seeds = sorted(common)
    if len(common_seeds) < 2:
        return {}
    seed_to_val = {
        m: dict(zip(method_seeds[m], method_values[m]))
        for m in method_values
        if m in method_seeds
    }
    return {
        m: [seed_to_val[m][s] for s in common_seeds]
        for m in method_values
        if m in seed_to_val and all(s in seed_to_val[m] for s in common_seeds)
    }


def compute_all_statistics(
    raw_results: Dict[str, Dict[str, List[float]]],
    method_seeds: Optional[Dict[str, List[int]]] = None,
    baseline_key: str = 'full_ft'
) -> Dict[str, Any]:
    """
    Compute statistics for all methods and metrics.
    When method_seeds is provided, comparisons use only seeds present for all methods
    (so paired tests work even if some runs are missing).
    
    Returns:
        Comprehensive statistics dictionary
    """
    stats_output = {
        'methods': {},
        'comparisons': {},
        'assumptions': {},
    }
    if method_seeds is None:
        method_seeds = {}

    # Get all metric names
    all_metrics = set()
    for method_data in raw_results.values():
        all_metrics.update(method_data.keys())

    # For each metric, compute statistics across methods
    for metric_name in all_metrics:
        method_values = {}
        for method_key, method_data in raw_results.items():
            if metric_name in method_data and method_data[metric_name]:
                method_values[method_key] = method_data[metric_name]

        if not method_values:
            continue

        # Compute statistics for each method (on all available seeds)
        for method_key, values in method_values.items():
            if method_key not in stats_output['methods']:
                stats_output['methods'][method_key] = {}
            stats_output['methods'][method_key][metric_name] = compute_statistics(values)
            stats_output['methods'][method_key][metric_name]['values'] = values

        # Perform comprehensive comparison on aligned seeds only
        if baseline_key in method_values and len(method_values) > 1:
            aligned = _align_by_common_seeds(method_values, method_seeds) if method_seeds else method_values
            if aligned and len(aligned) > 1 and len(next(iter(aligned.values()))) >= 2:
                comparison_results = perform_comprehensive_comparison(
                    aligned,
                    baseline_key=baseline_key
                )
                stats_output['comparisons'][metric_name] = comparison_results
            else:
                common = set.intersection(*(set(method_seeds.get(m, [])) for m in method_values)) if method_seeds else set()
                stats_output['comparisons'][metric_name] = {
                    'note': f'Skipped paired comparison (need ≥2 common seeds; common n={len(common)})'
                }

        # Check assumptions (on aligned data if available)
        if len(method_values) >= 2:
            check_vals = _align_by_common_seeds(method_values, method_seeds) if method_seeds else method_values
            if len(check_vals) >= 2 and all(len(v) >= 2 for v in check_vals.values()):
                assumptions = check_assumptions(check_vals)
                stats_output['assumptions'][metric_name] = assumptions

    return stats_output


def generate_summary_report(stats_output: Dict[str, Any]) -> str:
    """Generate human-readable summary report."""
    report = []
    report.append("=" * 80)
    report.append("MULTI-SEED STATISTICAL ANALYSIS SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    methods = stats_output.get('methods', {})
    if not methods:
        report.append("No method results to summarize.")
        return "\n".join(report)

    # Validation accuracy summary
    if 'val_accuracy' in next(iter(methods.values()), {}):
        report.append("VALIDATION ACCURACY:")
        report.append("-" * 80)
        
        for method_key, method_stats in methods.items():
            if 'val_accuracy' not in method_stats:
                continue
            stats = method_stats['val_accuracy']
            mean = stats.get('mean')
            if mean is None:
                continue
            mean = mean * 100 if mean < 2 else mean
            std = stats.get('std') or 0
            std = std * 100 if std < 2 else std
            ci_l = stats.get('ci_95_lower')
            ci_u = stats.get('ci_95_upper')
            if ci_l is not None:
                ci_l = ci_l * 100 if ci_l < 2 else ci_l
            if ci_u is not None:
                ci_u = ci_u * 100 if ci_u < 2 else ci_u
            n = stats.get('n', 0)
            report.append(f"{method_key:20s}: {mean:6.2f} ± {std:5.2f}%  "
                        f"95% CI: [{ci_l or 0:6.2f}, {ci_u or 0:6.2f}]  "
                        f"n={n}")
            # Add significance if available
            if stats.get('t_test'):
                p_val = stats['t_test'].get('p_value')
                if p_val is not None:
                    cohens_d = stats.get('cohens_d', np.nan)
                    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    report.append(f"{'':20s}  vs baseline: p={p_val:.4f} ({sig_marker}), "
                                f"Cohen's d={cohens_d:.3f}")
        report.append("")
    
    # Assumptions check
    if 'val_accuracy' in stats_output.get('assumptions', {}):
        report.append("STATISTICAL ASSUMPTIONS:")
        report.append("-" * 80)
        
        assumptions = stats_output['assumptions']['val_accuracy']
        
        for method_key, method_assumptions in assumptions.items():
            if method_key == 'homogeneity_of_variance':
                levene_p = method_assumptions['levene']['p_value']
                is_hom = "Yes" if method_assumptions['is_homogeneous'] else "No"
                report.append(f"Homogeneity of variance: {is_hom} (Levene p={levene_p:.4f})")
            elif 'normality' in method_assumptions:
                shapiro_p = method_assumptions['normality']['p_value']
                is_norm = "Yes" if method_assumptions['is_normal'] else "No"
                report.append(f"{method_key:20s}: Normal={is_norm} (Shapiro-Wilk p={shapiro_p:.4f})")
        
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Main function."""
    args = parse_args()
    
    print(f"Collecting results from: {args.base_dir}")
    print(f"Task: {args.task}")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}")
    print("")
    
    # Parse seeds
    seeds = parse_seed_spec(args.seeds)
    print(f"Parsed seeds: {seeds} (n={len(seeds)})")
    print("")
    
    # Collect raw results
    base_dir = Path(args.base_dir)
    raw_results, method_seeds = collect_multi_seed_results(base_dir, args.task, args.methods, seeds)

    print(f"\nCollected results for {len(raw_results)} methods")
    for method, metrics in raw_results.items():
        n_seeds = len(method_seeds.get(method, []))
        print(f"  {method}: {len(metrics)} metrics, {n_seeds} seeds")
        for metric_name, values in metrics.items():
            print(f"    - {metric_name}: {len(values)} values")

    # Compute statistics (comparisons aligned by common seeds)
    print("\nComputing statistics...")
    stats_output = compute_all_statistics(raw_results, method_seeds=method_seeds, baseline_key=args.baseline)
    
    # Add metadata
    stats_output['metadata'] = {
        'task': args.task,
        'num_seeds': len(seeds),
        'seeds': seeds,
        'methods': args.methods,
        'baseline': args.baseline,
    }
    
    # Generate summary report
    summary = generate_summary_report(stats_output)
    print("\n" + summary)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(stats_output, f, indent=2)
    
    print(f"\nSaved statistics to: {output_path}")
    
    # Also save summary report
    summary_path = output_path.with_suffix('.summary.txt')
    with open(summary_path, "w") as f:
        f.write(summary)
    
    print(f"Saved summary report to: {summary_path}")


if __name__ == "__main__":
    main()
