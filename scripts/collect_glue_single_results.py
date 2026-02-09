#!/usr/bin/env python3
"""Collect token-level accuracy from complete glue_single runs and emit MD + LaTeX."""

import json
import os
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs" / "glue_single"
DOC = Path(__file__).resolve().parent.parent / "doc"


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "evaluation_history.json").exists() or (run_dir / "training_summary.json").exists()


def parse_run_dir(name: str) -> Optional[Tuple[str, str, int]]:
    """Return (task, method, seed) or None if not matched."""
    if not name.endswith("_frozen"):
        return None
    base = name[:-7]  # drop _frozen
    if "_seed" not in base:
        return None
    task_method, seed_str = base.rsplit("_seed", 1)
    try:
        seed = int(seed_str)
    except ValueError:
        return None
    # task_method is e.g. sst2_full_ft or qnli_lora_diffusion
    parts = task_method.split("_")
    if len(parts) < 2:
        return None
    task = parts[0]
    method = "_".join(parts[1:])
    return (task, method, seed)


def get_metrics(run_dir: Path) -> Optional[dict]:
    """Return dict with best_token_acc, final_token_acc (as decimals 0--1), and optional best_step."""
    ts_path = run_dir / "training_summary.json"
    eh_path = run_dir / "evaluation_history.json"
    out = {}

    if ts_path.exists():
        with open(ts_path) as f:
            ts = json.load(f)
        out["best_token_acc"] = ts.get("best_metric")
        fe = ts.get("final_eval_metrics") or {}
        out["final_token_acc"] = fe.get("accuracy")
        out["final_gen_acc"] = fe.get("generation_accuracy")
        out["source"] = "training_summary"
        return out

    if eh_path.exists():
        with open(eh_path) as f:
            eh = json.load(f)
        if not eh:
            return None
        accs = [e["metrics"].get("accuracy") for e in eh if "accuracy" in e.get("metrics", {})]
        if not accs:
            return None
        out["best_token_acc"] = max(accs)
        out["final_token_acc"] = accs[-1]
        out["final_gen_acc"] = eh[-1].get("metrics", {}).get("generation_accuracy")
        best_idx = accs.index(max(accs))
        out["best_step"] = eh[best_idx].get("step")
        out["source"] = "evaluation_history"
        return out

    return None


def main():
    runs = []
    for d in sorted(OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        if not is_complete(d):
            continue
        parsed = parse_run_dir(d.name)
        if not parsed:
            continue
        task, method, seed = parsed
        metrics = get_metrics(d)
        if not metrics or metrics.get("best_token_acc") is None:
            continue
        runs.append({
            "task": task,
            "method": method,
            "seed": seed,
            "best_token_acc": metrics["best_token_acc"],
            "final_token_acc": metrics.get("final_token_acc"),
            "final_gen_acc": metrics.get("final_gen_acc"),
            "run_dir": d.name,
        })

    # Sort by task, method, seed
    runs.sort(key=lambda x: (x["task"], x["method"], x["seed"]))

    # --- Markdown file ---
    md_path = DOC / "GLUE_SINGLE_TASK_RESULTS.md"
    with open(md_path, "w") as f:
        f.write("# GLUE Single-Task Results (Token-Level Accuracy)\n\n")
        f.write("Collected from **{}** complete runs in `outputs/glue_single/`.\n\n".format(len(runs)))
        f.write("Metric: **token-level denoising accuracy** (fraction of masked tokens predicted correctly on the validation set).\n\n")
        f.write("| Task | Method | Seed | Best Val Acc (%) | Final Val Acc (%) |\n")
        f.write("|------|--------|------|------------------|-------------------|\n")
        for r in runs:
            best_pct = r["best_token_acc"] * 100 if r["best_token_acc"] <= 1 else r["best_token_acc"]
            final_pct = (r["final_token_acc"] * 100 if r["final_token_acc"] is not None and r["final_token_acc"] <= 1 else (r["final_token_acc"] if r["final_token_acc"] is not None else "---"))
            f.write("| {} | {} | {} | {:.2f} | {} |\n".format(
                r["task"].upper(),
                r["method"],
                r["seed"],
                best_pct,
                final_pct if isinstance(final_pct, str) else "{:.2f}".format(final_pct),
            ))
        f.write("\n## Summary by (Task, Method)\n\n")
        # Mean and std per (task, method)
        by_task_method = defaultdict(list)
        for r in runs:
            by_task_method[(r["task"], r["method"])].append(r["best_token_acc"] * 100 if r["best_token_acc"] <= 1 else r["best_token_acc"])
        f.write("| Task | Method | Mean (%) | Std (%) | N seeds |\n")
        f.write("|------|--------|----------|---------|--------|\n")
        for (task, method), accs in sorted(by_task_method.items()):
            mean = statistics.mean(accs)
            std = statistics.stdev(accs) if len(accs) > 1 else 0.0
            f.write("| {} | {} | {:.2f} | {:.2f} | {} |\n".format(task.upper(), method, mean, std, len(accs)))
        f.write("\n## Note on 100% token-level accuracy (QNLI, MRPC)\n\n")
        f.write("Token-level accuracy is **denoising accuracy**: at evaluation we mask the label token and measure whether the model predicts it correctly given the instruction (teacher-forced). For binary classification with a single-token label this can reach 100% and is **not a bug**. The more comparable metric is **generation accuracy** (model generates the label from scratch; decoded output vs reference). Below is the mean generation accuracy (%) where available.\n\n")
        by_task_method_gen = defaultdict(list)
        for r in runs:
            g = r.get("final_gen_acc")
            if g is not None:
                by_task_method_gen[(r["task"], r["method"])].append(g * 100 if g <= 1 else g)
        if by_task_method_gen:
            f.write("| Task | Method | Gen. acc. mean (%) | N seeds |\n")
            f.write("|------|--------|--------------------|--------|\n")
            for (task, method), accs in sorted(by_task_method_gen.items()):
                mean = statistics.mean(accs)
                f.write("| {} | {} | {:.2f} | {} |\n".format(task.upper(), method, mean, len(accs)))
        f.write("\n*Generated by `scripts/collect_glue_single_results.py`.*\n")

    print("Wrote", md_path)

    # --- LaTeX snippet for paper (summary table: mean ± std per task × method) ---
    by_task_method = defaultdict(list)
    for r in runs:
        by_task_method[(r["task"], r["method"])].append(r["best_token_acc"] * 100 if r["best_token_acc"] <= 1 else r["best_token_acc"])

    tasks_order = ["sst2", "qnli", "mrpc"]
    methods_order = ["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit"]
    method_display = {"full_ft": "Full Fine-Tuning", "lora_diffusion": "LoRA-Diffusion", "weight_lora": "Weight LoRA", "adapters": "Adapters", "bitfit": "BitFit"}

    latex_path = DOC / "glue_single_table.tex"
    with open(latex_path, "w") as f:
        f.write("% Summary table: mean $\\pm$ std over seeds (token-level val acc, \\%)\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Single-task GLUE results: token-level validation accuracy (\\%, mean $\\pm$ std over seeds). ")
        f.write("Based on %d complete runs. Token-level = denoising accuracy (predicting the masked label token given the instruction); for binary QNLI/MRPC this can reach 100\\%% and is not a bug. Generation accuracy (decoded output) is lower; see Table~\\ref{tab:glue_single_gen}.}\n" % len(runs))
        f.write("\\label{tab:glue_single_summary}\n")
        f.write("\\scriptsize\n")
        f.write("\\begin{tabular}{@{}l")
        for _ in tasks_order:
            f.write("c")
        f.write("@{}}\n\\toprule\n")
        f.write("Method & " + " & ".join(t.upper() for t in tasks_order) + " \\\\\n\\midrule\n")
        for method in methods_order:
            row = [method_display.get(method, method)]
            for task in tasks_order:
                accs = by_task_method.get((task, method), [])
                if accs:
                    mean = statistics.mean(accs)
                    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
                    row.append("${:.2f} \\pm {:.2f}$".format(mean, std))
                else:
                    row.append("---")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print("Wrote", latex_path)

    # --- Generation accuracy table (realistic metric for QNLI/MRPC) ---
    by_task_method_gen = defaultdict(list)
    for r in runs:
        g = r.get("final_gen_acc")
        if g is not None:
            by_task_method_gen[(r["task"], r["method"])].append(g * 100 if g <= 1 else g)
    gen_tex_path = DOC / "glue_single_gen_table.tex"
    with open(gen_tex_path, "w") as f:
        f.write("% Generation accuracy (decoded output vs reference)\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Generation accuracy (\\%, mean $\\pm$ std over seeds): model generates the label from scratch; decoded output compared to reference. More comparable to standard GLUE than token-level denoising.}\n")
        f.write("\\label{tab:glue_single_gen}\n")
        f.write("\\scriptsize\n")
        f.write("\\begin{tabular}{@{}lccc@{}}\n\\toprule\n")
        f.write("Method & SST2 & QNLI & MRPC \\\\\n\\midrule\n")
        for method in methods_order:
            row = [method_display.get(method, method)]
            for task in tasks_order:
                accs = by_task_method_gen.get((task, method), [])
                if accs:
                    mean = statistics.mean(accs)
                    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
                    row.append("${:.2f} \\pm {:.2f}$".format(mean, std))
                else:
                    row.append("---")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("Wrote", gen_tex_path)

    # Also write full per-run table LaTeX (for appendix or supplementary)
    full_tex = DOC / "glue_single_full_table.tex"
    with open(full_tex, "w") as f:
        f.write("% Full per-run table (57 runs)\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{All single-task runs: token-level validation accuracy (\\%). }\n")
        f.write("\\label{tab:glue_single_full}\n")
        f.write("\\tiny\n")
        f.write("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n")
        f.write("Task & Method & Seed & Best Val (\\%) & Final Val (\\%) \\\\\n\\midrule\n")
        for r in runs:
            best_pct = r["best_token_acc"] * 100 if r["best_token_acc"] <= 1 else r["best_token_acc"]
            final = r["final_token_acc"]
            final_str = "---" if final is None else "{:.2f}".format(final * 100 if final <= 1 else final)
            f.write("{} & {} & {} & {:.2f} & {} \\\\\n".format(
                r["task"].upper(), r["method"], r["seed"], best_pct, final_str
            ))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print("Wrote", full_tex)

    return runs


if __name__ == "__main__":
    main()
