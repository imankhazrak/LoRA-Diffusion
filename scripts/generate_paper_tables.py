#!/usr/bin/env python3
"""Generate comprehensive tables for paper from training results with statistics."""

import argparse
import json
import glob
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.statistical_analysis import format_stats_for_table, get_significance_marker


def load_json(filepath: Path) -> Any:
    """Load JSON file."""
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return {}


def extract_method_name(output_dir: str) -> str:
    """Extract method name from output directory path."""
    parts = Path(output_dir).name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:-1])
    return "unknown"


def format_number(value: Optional[float], decimals: int = 2, percentage: bool = False) -> str:
    """Format number for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "---"
    if percentage:
        return f"{value * 100:.{decimals}f}\\%"
    return f"{value:.{decimals}f}"


def collect_comprehensive_results() -> pd.DataFrame:
    """Collect comprehensive results from all runs."""
    results = []
    output_dirs = sorted(glob.glob("outputs/sst2_*_*"))
    
    for output_dir in output_dirs:
        output_path = Path(output_dir)
        method = extract_method_name(output_dir)
        job_id = output_path.name.split("_")[-1]
        
        # Load data
        summary = load_json(output_path / "training_summary.json")
        training_history = load_json(output_path / "training_history.json")
        eval_history = load_json(output_path / "evaluation_history.json")
        
        # Get checkpoint info
        checkpoint_dir = Path("checkpoints") / output_path.name / "final_model"
        metadata = load_json(checkpoint_dir / "checkpoint_metadata.json")
        
        # Extract metrics
        final_train = summary.get("final_training_metrics", {})
        final_eval = summary.get("final_eval_metrics", {})
        
        # Find best evaluation metrics
        best_eval_acc = 0.0
        best_eval_step = 0
        best_eval_loss = None
        if eval_history:
            for entry in eval_history:
                metrics = entry.get("metrics", {})
                acc = metrics.get("accuracy", 0.0)
                if acc > best_eval_acc:
                    best_eval_acc = acc
                    best_eval_step = entry.get("step", 0)
                    best_eval_loss = metrics.get("loss")
        
        # Calculate convergence metrics
        steps_to_70 = None
        steps_to_80 = None
        if training_history:
            for entry in training_history:
                acc = entry.get("metrics", {}).get("accuracy", 0.0)
                step = entry.get("step", 0)
                if steps_to_70 is None and acc >= 0.70:
                    steps_to_70 = step
                if steps_to_80 is None and acc >= 0.80:
                    steps_to_80 = step
        
        # Parameter counts
        total_params = metadata.get("total_parameters", 0)
        trainable_params = metadata.get("trainable_parameters", 0)
        param_ratio = (trainable_params / total_params * 100) if total_params > 0 else 0.0
        
        # Method display name
        method_display = {
            "lora_diffusion": "LoRA-Diffusion",
            "full_ft": "Full Fine-tuning",
            "bitfit": "BitFit",
            "weight_lora": "Weight LoRA",
            "adapters": "Adapters",
            "prefix_tuning": "Prefix Tuning",
        }.get(method, method.replace("_", "-").title())
        
        results.append({
            "Method": method_display,
            "Method Code": method,
            "Job ID": job_id,
            "Total Steps": summary.get("total_steps", 0),
            "Final Train Loss": final_train.get("loss"),
            "Final Train Acc (%)": final_train.get("accuracy", 0.0) * 100 if final_train.get("accuracy") else None,
            "Best Eval Acc (%)": best_eval_acc * 100 if best_eval_acc > 0 else None,
            "Best Eval Loss": best_eval_loss,
            "Best Eval Step": best_eval_step if best_eval_step > 0 else None,
            "Steps to 70%": steps_to_70,
            "Steps to 80%": steps_to_80,
            "Trainable Params (M)": trainable_params / 1e6 if trainable_params > 0 else None,
            "Total Params (M)": total_params / 1e6 if total_params > 0 else None,
            "Param Ratio (%)": param_ratio,
        })
    
    return pd.DataFrame(results)


def generate_latex_table(df: pd.DataFrame, caption: str = "Training Results Comparison", label: str = "tab:results") -> str:
    """Generate LaTeX table."""
    # Select columns for main table
    display_cols = [
        "Method",
        "Final Train Acc (%)",
        "Best Eval Acc (%)",
        "Steps to 70%",
        "Trainable Params (M)",
        "Param Ratio (%)",
    ]
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{l" + "c" * (len(display_cols) - 1) + "}\n"
    latex += "\\toprule\n"
    
    # Headers
    headers = {
        "Method": "Method",
        "Final Train Acc (%)": "Train Acc. (\\%)",
        "Best Eval Acc (%)": "Eval Acc. (\\%)",
        "Steps to 70%": "Steps to 70\\%",
        "Trainable Params (M)": "Trainable Params (M)",
        "Param Ratio (%)": "Param Ratio (\\%)",
    }
    latex += " & ".join([headers.get(col, col) for col in display_cols]) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Rows
    for _, row in df.iterrows():
        values = []
        for col in display_cols:
            val = row[col]
            if col == "Method":
                values.append(val)
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                values.append("---")
            elif isinstance(val, float):
                if "Params" in col:
                    values.append(f"{val:.2f}")
                else:
                    values.append(f"{val:.2f}")
            else:
                values.append(str(int(val)))
        latex += " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    return latex


def generate_detailed_table(df: pd.DataFrame) -> str:
    """Generate detailed markdown table."""
    return df.to_markdown(index=False)


def main():
    print("Collecting results from all training runs...\n")
    
    df = collect_comprehensive_results()
    
    if df.empty:
        print("No results found. Please ensure training has completed.")
        return
    
    # Sort by method
    method_order = ["LoRA-Diffusion", "Full Fine-tuning", "Weight LoRA", "Adapters", "BitFit", "Prefix Tuning"]
    df["Method_Order"] = df["Method"].apply(lambda x: method_order.index(x) if x in method_order else 999)
    df = df.sort_values("Method_Order").drop("Method_Order", axis=1)
    
    # Print summary
    print("=" * 100)
    print("COMPREHENSIVE TRAINING RESULTS")
    print("=" * 100 + "\n")
    print(df.to_string(index=False))
    print("\n")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    with open("paper_results_table.tex", "w") as f:
        f.write(latex_table)
    print("Generated: paper_results_table.tex\n")
    print("LaTeX Table Preview:")
    print("-" * 80)
    print(latex_table)
    print("-" * 80 + "\n")
    
    # Generate detailed markdown
    with open("paper_results_detailed.md", "w") as f:
        f.write("# Detailed Training Results\n\n")
        f.write(generate_detailed_table(df))
    print("Generated: paper_results_detailed.md\n")
    
    # Save CSV
    df.to_csv("paper_results.csv", index=False)
    print("Generated: paper_results.csv\n")
    
    # Save JSON
    df.to_json("paper_results.json", orient="records", indent=2)
    print("Generated: paper_results.json\n")
    
    # Summary statistics
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100 + "\n")
    
    # Filter to only completed runs (with actual results)
    completed_df = df[df["Total Steps"] > 0].copy()
    
    if not completed_df.empty:
        if "Final Train Acc (%)" in completed_df.columns:
            valid_acc = completed_df["Final Train Acc (%)"].dropna()
            if not valid_acc.empty:
                best_idx = valid_acc.idxmax()
                print(f"Best Training Accuracy: {valid_acc.max():.2f}% ({completed_df.loc[best_idx, 'Method']})")
                print(f"Average Training Accuracy: {valid_acc.mean():.2f}%")
        
        if "Param Ratio (%)" in completed_df.columns:
            valid_ratio = completed_df["Param Ratio (%)"].dropna()
            if not valid_ratio.empty and (valid_ratio > 0).any():
                min_idx = valid_ratio[valid_ratio > 0].idxmin()
                print(f"Most Parameter-Efficient: {completed_df.loc[min_idx, 'Method']} ({valid_ratio[valid_ratio > 0].min():.2f}%)")
        
        print(f"\nCompleted Runs: {len(completed_df)}")
        print(f"Total Runs Found: {len(df)}")
    else:
        print("No completed runs found with results.")
    print()
    return
    
    # New path: generate tables from statistics
    print(f"Generating tables from: {stats_file}\n")
    
    tables = generate_tables_from_stats(stats_file)
    
    # Write all tables to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for table_name, table_latex in tables.items():
            f.write(f"% {table_name}\n")
            f.write(table_latex)
            f.write("\n\n")
    
    print(f"Generated {len(tables)} tables:")
    for table_name in tables.keys():
        print(f"  - {table_name}")
    
    print(f"\nSaved to: {output_path}")
    
    # Print preview
    print("\n" + "=" * 80)
    print("TABLE PREVIEW")
    print("=" * 80 + "\n")
    
    for table_name, table_latex in tables.items():
        print(f"--- {table_name} ---")
        print(table_latex)
        print()


if __name__ == "__main__":
    main()
