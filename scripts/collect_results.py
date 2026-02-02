#!/usr/bin/env python3
"""Collect experiment results and prepare data for paper table updates."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_parameter_counts(counts_file: Path) -> Dict[str, Any]:
    """Load parameter counts from JSON file."""
    with open(counts_file, "r") as f:
        return json.load(f)


def find_experiment_outputs(base_dir: Path, task: str, job_id: str) -> Dict[str, Path]:
    """Find output directories for each method."""
    outputs = {}
    methods = ["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit"]
    
    for method in methods:
        pattern = f"{task}_{method}_seed*_{job_id}"
        matches = list(base_dir.glob(pattern))
        if matches:
            outputs[method] = matches[0]
        else:
            # Try alternative patterns
            alt_patterns = [
                f"{task}_{method}_*{job_id}*",
                f"*{task}*{method}*{job_id}*",
            ]
            for alt_pattern in alt_patterns:
                matches = list(base_dir.glob(alt_pattern))
                if matches:
                    outputs[method] = matches[0]
                    break
    
    return outputs


def extract_training_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract training metrics from output directory."""
    metrics = {}
    
    # Try to find training history
    history_file = output_dir / "training_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
            if history:
                # Get initial and final metrics
                initial_entry = history[0]
                final_entry = history[-1]
                metrics["initial_loss"] = initial_entry.get("metrics", {}).get("loss", None)
                metrics["final_loss"] = final_entry.get("metrics", {}).get("loss", None)
                metrics["final_train_acc"] = final_entry.get("metrics", {}).get("accuracy", None)
                metrics["steps"] = final_entry.get("step", None)
    
    # Try to find training summary (has final metrics)
    summary_file = output_dir / "training_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
            if "final_training_metrics" in summary:
                final_train = summary["final_training_metrics"]
                if metrics.get("final_loss") is None:
                    metrics["final_loss"] = final_train.get("loss")
                if metrics.get("final_train_acc") is None:
                    metrics["final_train_acc"] = final_train.get("accuracy")
            if "total_steps" in summary and metrics.get("steps") is None:
                metrics["steps"] = summary["total_steps"]
    
    # Try to find eval results (from separate evaluation script - has task-specific metrics)
    eval_results_file = output_dir / "eval_results.json"
    if eval_results_file.exists():
        with open(eval_results_file, "r") as f:
            eval_data = json.load(f)
            eval_metrics = eval_data.get("metrics", {})
            # Task-specific accuracy (from text generation)
            metrics["val_accuracy"] = eval_metrics.get("accuracy", None)
            metrics["val_f1"] = eval_metrics.get("f1", None)
            metrics["val_loss"] = eval_data.get("metrics", {}).get("loss", None)
    
    # Try to find evaluation history (during training - might be token-level)
    eval_history_file = output_dir / "evaluation_history.json"
    if eval_history_file.exists() and metrics.get("val_accuracy") is None:
        with open(eval_history_file, "r") as f:
            eval_history = json.load(f)
            if eval_history:
                # Get best validation metrics (but note: these might be token-level, not task-level)
                best_eval = max(eval_history, key=lambda x: x.get("metrics", {}).get("accuracy", 0))
                # Only use if it looks like a reasonable task accuracy (>0.1), not token-level (<0.1)
                token_acc = best_eval.get("metrics", {}).get("accuracy", 0)
                if token_acc > 0.1:  # Likely task-level accuracy
                    metrics["val_accuracy"] = token_acc
                metrics["val_loss"] = best_eval.get("metrics", {}).get("loss", None)
    
    return metrics if metrics else None


def collect_all_results(
    parameter_counts_file: Path,
    base_output_dir: Path,
    task: str,
    job_id: str,
) -> Dict[str, Any]:
    """Collect all results from experiments."""
    results = {}
    
    # Load parameter counts
    param_counts = load_parameter_counts(parameter_counts_file)
    results["parameter_counts"] = param_counts
    
    # Find experiment outputs
    output_dirs = find_experiment_outputs(base_output_dir, task, job_id)
    results["experiment_outputs"] = {}
    
    for method, output_dir in output_dirs.items():
        metrics = extract_training_metrics(output_dir)
        results["experiment_outputs"][method] = {
            "output_dir": str(output_dir),
            "metrics": metrics,
        }
    
    return results


def format_for_paper_tables(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format results for paper table updates."""
    param_counts = results["parameter_counts"]
    exp_outputs = results["experiment_outputs"]
    
    formatted = {}
    
    for method_key in ["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit", "prefix_tuning"]:
        method_data = {}
        
        # Parameter data
        if method_key in param_counts:
            pc = param_counts[method_key]
            method_data["trainable_params"] = pc["trainable_params"]
            method_data["trainable_percent"] = pc["trainable_percent"]
            method_data["storage_mb"] = pc["storage_mb"]
            method_data["base_params"] = pc["base_params"]
        
        # Experiment data
        if method_key in exp_outputs:
            exp = exp_outputs[method_key]
            if exp.get("metrics"):
                metrics = exp["metrics"]
                method_data["val_accuracy"] = metrics.get("val_accuracy")
                method_data["train_loss"] = metrics.get("final_loss")
                method_data["initial_loss"] = metrics.get("initial_loss")
                method_data["steps"] = metrics.get("steps")
                method_data["final_train_acc"] = metrics.get("final_train_acc")
        
        formatted[method_key] = method_data
    
    return formatted


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument(
        "--parameter-counts",
        type=str,
        default="parameter_counts.json",
        help="Path to parameter_counts.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (will search for experiment outputs)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sst2",
        help="Task name",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="SLURM job ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="collected_results.json",
        help="Output JSON file",
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        # Try common locations
        scratch = Path(f"/fs/scratch/users/{Path.home().name}/lora-diffusion/outputs")
        local = Path("outputs")
        if scratch.exists():
            base_output_dir = scratch
        elif local.exists():
            base_output_dir = local
        else:
            base_output_dir = Path("outputs")
    
    param_counts_file = Path(args.parameter_counts)
    if not param_counts_file.exists():
        print(f"ERROR: Parameter counts file not found: {param_counts_file}")
        sys.exit(1)
    
    # Collect results
    print(f"Collecting results for job {args.job_id}...")
    print(f"Parameter counts: {param_counts_file}")
    print(f"Output directory: {base_output_dir}")
    
    results = collect_all_results(
        parameter_counts_file=param_counts_file,
        base_output_dir=base_output_dir,
        task=args.task,
        job_id=args.job_id,
    )
    
    # Format for paper tables
    formatted = format_for_paper_tables(results)
    
    # Save results
    output_file = Path(args.output)
    with open(output_file, "w") as f:
        json.dump({
            "raw_results": results,
            "formatted_for_tables": formatted,
        }, f, indent=2)
    
    print(f"\nResults collected and saved to: {output_file}")
    print("\nFormatted results for paper tables:")
    print(json.dumps(formatted, indent=2))
    
    return formatted


if __name__ == "__main__":
    main()
