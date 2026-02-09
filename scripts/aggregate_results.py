#!/usr/bin/env python3
"""Aggregate run results: results_summary.csv, results_summary.json, LaTeX table snippets (Instruction2.md)."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.collect_results_with_stats import (
    find_experiment_output,
    extract_metrics_from_output,
    collect_multi_seed_results,
    parse_seed_spec,
    compute_all_statistics,
)


def load_json(p: Path) -> Any:
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def aggregate_per_task(
    base_dir: Path,
    task: str,
    methods: List[str],
    seeds: List[int],
    baseline: str = "full_ft",
) -> Dict[str, Any]:
    raw_results, method_seeds = collect_multi_seed_results(base_dir, task, methods, seeds)
    stats = compute_all_statistics(raw_results, method_seeds=method_seeds, baseline_key=baseline)
    return {"task": task, "raw": raw_results, "method_seeds": method_seeds, "stats": stats}


def stats_to_row(metric_name: str, method: str, stats: Dict) -> Dict[str, Any]:
    mean = stats.get("mean")
    std = stats.get("std")
    ci_l = stats.get("ci_95_lower")
    ci_u = stats.get("ci_95_upper")
    n = stats.get("n", 0)
    return {
        "metric": metric_name,
        "method": method,
        "mean": mean,
        "std": std,
        "ci_95_lower": ci_l,
        "ci_95_upper": ci_u,
        "n": n,
    }


def write_csv(all_rows: List[Dict], out_path: Path):
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)


def write_latex_token_accuracy(by_task: Dict[str, Dict], out_path: Path):
    lines = [
        "% Table: Token-level denoising accuracy (validation). Do not mix with GLUE accuracy.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Token-level denoising accuracy (validation, mean $\\pm$ std over seeds).}",
        "\\label{tab:token_accuracy}",
        "\\begin{tabular}{l" + "c" * len(by_task) + "}",
        "\\toprule",
        "Method & " + " & ".join(t.replace("_", "-").upper() for t in by_task) + " \\\\",
        "\\midrule",
    ]
    task0 = next(iter(by_task.values()))
    methods = list(task0.get("methods", {}).keys())
    for method in methods:
        method_display = method.replace("_", " ").title()
        cells = []
        for task in by_task:
            mdict = by_task[task].get("methods", {}).get(method, {})
            st = mdict.get("val_token_accuracy") or mdict.get("token_accuracy_val") or mdict.get("val_accuracy")
            if st and st.get("mean") is not None:
                m, s = st["mean"] * 100, (st.get("std") or 0) * 100
                cells.append(f"{m:.1f} $\\pm$ {s:.1f}")
            else:
                cells.append("---")
        lines.append(f"{method_display} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_path.write_text("\n".join(lines))


def write_latex_glue(by_task: Dict[str, Dict], out_path: Path):
    lines = [
        "% Table: Standard GLUE metrics (head-free, verbalizer). Accuracy and F1 (MRPC only).",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{GLUE task accuracy (and F1 for MRPC) from head-free decoding (mean $\\pm$ std).}",
        "\\label{tab:glue_metrics}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Task & Method & Accuracy & F1 \\\\",
        "\\midrule",
    ]
    for task in by_task:
        methods = by_task[task].get("methods", {})
        for method, metrics in methods.items():
            method_display = method.replace("_", " ").title()
            acc = metrics.get("glue_accuracy_val") or metrics.get("val_generation_accuracy") or metrics.get("val_accuracy")
            f1 = metrics.get("glue_f1_val") or metrics.get("val_f1")
            acc_str = f"{acc['mean']*100:.1f} $\\pm$ {(acc.get('std') or 0)*100:.1f}" if acc and acc.get("mean") is not None else "---"
            f1_str = f"{f1['mean']*100:.1f} $\\pm$ {(f1.get('std') or 0)*100:.1f}" if f1 and f1.get("mean") is not None else "---"
            lines.append(f"{task.upper()} & {method_display} & {acc_str} & {f1_str} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_path.write_text("\n".join(lines))


def _num_from_metric(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, dict):
        return v.get("mean") or 0.0
    return float(v)


def write_latex_params_efficiency(by_task: Dict[str, Dict], out_path: Path):
    lines = [
        "% Table: Parameter accounting and efficiency (PEFT-only and total trainable).",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Trainable parameters (PEFT-only and total) and checkpoint size.}",
        "\\label{tab:params_efficiency}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & PEFT-only (M) & Total trainable (M) & Checkpoint (MB) \\\\",
        "\\midrule",
    ]
    task0 = next(iter(by_task.values()))
    methods = task0.get("methods", {})
    for method in methods:
        method_display = method.replace("_", " ").title()
        mdict = task0["methods"][method]
        peft_m = _num_from_metric(mdict.get("trainable_params_peft_only")) / 1e6
        total_m = _num_from_metric(mdict.get("trainable_params_total")) / 1e6
        ckpt = mdict.get("checkpoint_mb")
        ckpt_str = f"{_num_from_metric(ckpt):.1f}" if ckpt is not None else "---"
        lines.append(f"{method_display} & {peft_m:.2f} & {total_m:.2f} & {ckpt_str} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="Aggregate results into CSV, JSON, and LaTeX tables")
    p.add_argument("--base_dir", type=str, default="./outputs/multi_seed_experiments", help="Base dir of run dirs")
    p.add_argument("--tasks", type=str, nargs="+", default=["sst2", "qnli", "mrpc"])
    p.add_argument("--methods", type=str, nargs="+", default=["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit"])
    p.add_argument("--seeds", type=str, default="42-46", help="Seed range or list")
    p.add_argument("--baseline", type=str, default="full_ft")
    p.add_argument("--output_dir", type=str, default=".", help="Where to write results_summary.* and *.tex")
    args = p.parse_args()
    seeds = parse_seed_spec(args.seeds)
    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_task = {}
    all_rows = []
    for task in args.tasks:
        ag = aggregate_per_task(base_dir, task, args.methods, seeds, args.baseline)
        by_task[task] = {"methods": ag["stats"].get("methods", {})}
        for method, metrics in ag["stats"].get("methods", {}).items():
            for metric_name, st in metrics.items():
                if metric_name == "values" or not isinstance(st, dict):
                    continue
                all_rows.append({"task": task, **stats_to_row(metric_name, method, st)})

    summary = {
        "tasks": args.tasks,
        "methods": args.methods,
        "seeds": seeds,
        "by_task": by_task,
    }
    csv_path = out_dir / "results_summary.csv"
    json_path = out_dir / "results_summary.json"
    write_csv(all_rows, csv_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {csv_path}, {json_path}")

    write_latex_token_accuracy(by_task, out_dir / "table_token_accuracy.tex")
    write_latex_glue(by_task, out_dir / "table_glue_metrics.tex")
    write_latex_params_efficiency(by_task, out_dir / "table_params_efficiency.tex")
    print(f"Wrote LaTeX snippets to {out_dir}")


if __name__ == "__main__":
    main()
