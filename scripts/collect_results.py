#!/usr/bin/env python3
"""Collect and format results from all training runs for paper tables."""

import json
import glob
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any
import argparse


def load_training_summary(output_dir: Path) -> Dict[str, Any]:
    """Load training summary from output directory."""
    summary_file = output_dir / "training_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return {}


def load_training_history(output_dir: Path) -> List[Dict]:
    """Load training history from output directory."""
    history_file = output_dir / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            return json.load(f)
    return []


def load_evaluation_history(output_dir: Path) -> List[Dict]:
    """Load evaluation history from output directory."""
    eval_file = output_dir / "evaluation_history.json"
    if eval_file.exists():
        with open(eval_file) as f:
            return json.load(f)
    return []


def extract_method_name(output_dir: str) -> str:
    """Extract method name from output directory path."""
    parts = Path(output_dir).name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:-1])  # Everything between task and job_id
    return "unknown"


def count_parameters(checkpoint_dir: Path) -> Dict[str, int]:
    """Count parameters from checkpoint metadata."""
    metadata_file = checkpoint_dir / "final_model" / "checkpoint_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            data = json.load(f)
            return {
                "total_params": data.get("total_parameters", 0),
                "trainable_params": data.get("trainable_parameters", 0),
            }
    return {"total_params": 0, "trainable_params": 0}


def collect_all_results(base_dir: str = "outputs") -> pd.DataFrame:
    """Collect all results from output directories."""
    results = []
    output_dirs = glob.glob(f"{base_dir}/sst2_*_*")
    
    for output_dir in sorted(output_dirs):
        output_path = Path(output_dir)
        method = extract_method_name(output_dir)
        job_id = output_path.name.split("_")[-1]
        
        # Load summary
        summary = load_training_summary(output_path)
        training_history = load_training_history(output_path)
        eval_history = load_evaluation_history(output_path)
        
        # Get checkpoint directory
        checkpoint_dir = Path("checkpoints") / output_path.name
        
        # Count parameters
        param_counts = count_parameters(checkpoint_dir)
        
        # Extract metrics
        final_metrics = summary.get("final_training_metrics", {})
        final_eval_metrics = summary.get("final_eval_metrics", {})
        
        # Get best metrics from evaluation history
        best_accuracy = 0.0
        best_step = 0
        if eval_history:
            for eval_entry in eval_history:
                metrics = eval_entry.get("metrics", {})
                acc = metrics.get("accuracy", 0.0)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_step = eval_entry.get("step", 0)
        
        # Calculate training efficiency (steps to reach 70% accuracy)
        steps_to_70 = None
        if training_history:
            for entry in training_history:
                if entry.get("metrics", {}).get("accuracy", 0.0) >= 0.70:
                    steps_to_70 = entry.get("step", None)
                    break
        
        results.append({
            "Method": method.replace("_", "-").title(),
            "Job ID": job_id,
            "Total Steps": summary.get("total_steps", 0),
            "Final Train Loss": round(final_metrics.get("loss", 0.0), 4) if final_metrics.get("loss") else None,
            "Final Train Acc": round(final_metrics.get("accuracy", 0.0) * 100, 2) if final_metrics.get("accuracy") else None,
            "Best Eval Acc": round(best_accuracy * 100, 2) if best_accuracy > 0 else None,
            "Best Eval Step": best_step if best_step > 0 else None,
            "Steps to 70%": steps_to_70 if steps_to_70 else None,
            "Trainable Params": param_counts.get("trainable_params", 0),
            "Total Params": param_counts.get("total_params", 0),
            "Param Ratio (%)": round(param_counts.get("trainable_params", 0) / max(param_counts.get("total_params", 1), 1) * 100, 2) if param_counts.get("total_params", 0) > 0 else None,
        })
    
    return pd.DataFrame(results)


def format_latex_table(df: pd.DataFrame) -> str:
    """Format DataFrame as LaTeX table."""
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Training Results Comparison}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\begin{tabular}{" + "l" + "c" * (len(df.columns) - 1) + "}\n"
    latex += "\\toprule\n"
    
    # Header
    headers = [col.replace("_", " ").title() for col in df.columns]
    latex += " & ".join(headers) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Rows
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                values.append("---")
            elif isinstance(val, float):
                values.append(f"{val:.2f}")
            else:
                values.append(str(val))
        latex += " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    return latex


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as Markdown table."""
    return df.to_markdown(index=False)


def main():
    parser = argparse.ArgumentParser(description="Collect and format training results")
    parser.add_argument("--format", choices=["latex", "markdown", "csv", "all"], default="all",
                       help="Output format")
    parser.add_argument("--output", type=str, default="results_summary",
                       help="Output file prefix (without extension)")
    
    args = parser.parse_args()
    
    # Collect results
    df = collect_all_results()
    
    if df.empty:
        print("No results found. Make sure training has completed.")
        return
    
    # Sort by method name
    df = df.sort_values("Method")
    
    # Print to console
    print("\n" + "=" * 80)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 80 + "\n")
    print(df.to_string(index=False))
    print("\n")
    
    # Save in requested formats
    if args.format in ["csv", "all"]:
        csv_file = f"{args.output}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved CSV: {csv_file}")
    
    if args.format in ["markdown", "all"]:
        md_file = f"{args.output}.md"
        with open(md_file, "w") as f:
            f.write("# Training Results Summary\n\n")
            f.write(format_markdown_table(df))
        print(f"Saved Markdown: {md_file}")
    
    if args.format in ["latex", "all"]:
        tex_file = f"{args.output}.tex"
        with open(tex_file, "w") as f:
            f.write(format_latex_table(df))
        print(f"Saved LaTeX: {tex_file}")
    
    # Create detailed results JSON
    json_file = f"{args.output}.json"
    df.to_json(json_file, orient="records", indent=2)
    print(f"Saved JSON: {json_file}")


if __name__ == "__main__":
    main()
