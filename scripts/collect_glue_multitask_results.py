#!/usr/bin/env python3
"""Collect metrics from complete glue_multitask (joint) runs and emit MD + LaTeX."""

import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs" / "glue_multitask"
DOC = Path(__file__).resolve().parent.parent / "doc"


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "evaluation_history.json").exists() or (run_dir / "training_summary.json").exists()


def parse_run_dir(name: str) -> Optional[Tuple[str, int]]:
    """Return (method, seed) or None. Expects multitask_{method}_seed{seed}_frozen."""
    if not name.endswith("_frozen"):
        return None
    base = name[:-7]
    if not base.startswith("multitask_") or "_seed" not in base:
        return None
    method_part = base[len("multitask_"):]
    method_part, seed_str = method_part.rsplit("_seed", 1)
    try:
        seed = int(seed_str)
    except ValueError:
        return None
    method = method_part
    return (method, seed)


def get_metrics(run_dir: Path) -> Optional[dict]:
    """Return dict with best_token_acc, final_token_acc, final_gen_acc (decimals 0--1)."""
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
        method, seed = parsed
        metrics = get_metrics(d)
        if not metrics or metrics.get("best_token_acc") is None:
            continue
        runs.append({
            "method": method,
            "seed": seed,
            "best_token_acc": metrics["best_token_acc"],
            "final_token_acc": metrics.get("final_token_acc"),
            "final_gen_acc": metrics.get("final_gen_acc"),
            "run_dir": d.name,
        })

    runs.sort(key=lambda x: (x["method"], x["seed"]))

    # --- Markdown file ---
    md_path = DOC / "GLUE_MULTITASK_RESULTS.md"
    with open(md_path, "w") as f:
        f.write("# GLUE Multi-Task (Joint) Results\n\n")
        f.write("Collected from **{}** complete runs in `outputs/glue_multitask/`.\n\n".format(len(runs)))
        f.write("Joint training on SST-2, QNLI, and MRPC. Metric: **token-level denoising accuracy** (and **generation accuracy**) on the combined validation set.\n\n")
        f.write("| Method | Seed | Best Val Acc (%) | Final Val Acc (%) | Gen Acc (%) |\n")
        f.write("|--------|------|------------------|-------------------|-------------|\n")
        for r in runs:
            best_pct = r["best_token_acc"] * 100 if r["best_token_acc"] <= 1 else r["best_token_acc"]
            final = r["final_token_acc"]
            final_pct = "---" if final is None else "{:.2f}".format(final * 100 if final <= 1 else final)
            gen = r.get("final_gen_acc")
            gen_pct = "---" if gen is None else "{:.2f}".format(gen * 100 if gen <= 1 else gen)
            f.write("| {} | {} | {:.2f} | {} | {} |\n".format(r["method"], r["seed"], best_pct, final_pct, gen_pct))
        f.write("\n## Summary by Method (mean ± std over seeds)\n\n")
        by_method = defaultdict(list)
        for r in runs:
            by_method[r["method"]].append(r["best_token_acc"] * 100 if r["best_token_acc"] <= 1 else r["best_token_acc"])
        f.write("| Method | Token-level mean (%) | Std (%) | N seeds |\n")
        f.write("|--------|----------------------|---------|--------|\n")
        for method in sorted(by_method.keys()):
            accs = by_method[method]
            mean = statistics.mean(accs)
            std = statistics.stdev(accs) if len(accs) > 1 else 0.0
            f.write("| {} | {:.2f} | {:.2f} | {} |\n".format(method, mean, std, len(accs)))
        by_method_gen = defaultdict(list)
        for r in runs:
            g = r.get("final_gen_acc")
            if g is not None:
                by_method_gen[r["method"]].append(g * 100 if g <= 1 else g)
        if by_method_gen:
            f.write("\n| Method | Gen. acc. mean (%) | Std (%) | N seeds |\n")
            f.write("|--------|--------------------|---------|--------|\n")
            for method in sorted(by_method_gen.keys()):
                accs = by_method_gen[method]
                mean = statistics.mean(accs)
                std = statistics.stdev(accs) if len(accs) > 1 else 0.0
                f.write("| {} | {:.2f} | {:.2f} | {} |\n".format(method, mean, std, len(accs)))
        f.write("\n*Generated by `scripts/collect_glue_multitask_results.py`.*\n")

    print("Wrote", md_path)

    # --- LaTeX table: Method | Token-level mean±std | Gen acc mean±std ---
    methods_order = ["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit"]
    method_display = {"full_ft": "Full Fine-Tuning", "lora_diffusion": "LoRA-Diffusion", "weight_lora": "Weight LoRA", "adapters": "Adapters", "bitfit": "BitFit"}

    by_method = defaultdict(list)
    for r in runs:
        by_method[r["method"]].append(r["best_token_acc"] * 100 if r["best_token_acc"] <= 1 else r["best_token_acc"])
    by_method_gen = defaultdict(list)
    for r in runs:
        g = r.get("final_gen_acc")
        if g is not None:
            by_method_gen[r["method"]].append(g * 100 if g <= 1 else g)

    latex_path = DOC / "glue_multitask_table.tex"
    with open(latex_path, "w") as f:
        f.write("% Multi-task joint training: mean $\\pm$ std over seeds\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{{Multi-task GLUE results (joint training on SST-2, QNLI, MRPC): token-level validation accuracy and generation accuracy (\\%, mean $\\pm$ std over seeds). Based on {} complete runs.}}\n".format(len(runs)))
        f.write("\\label{tab:glue_multitask}\n")
        f.write("\\scriptsize\n")
        f.write("\\begin{tabular}{@{}lcc@{}}\n\\toprule\n")
        f.write("Method & Token-level acc (\\%) & Gen. acc (\\%) \\\\\n\\midrule\n")
        for method in methods_order:
            accs = by_method.get(method, [])
            gen_accs = by_method_gen.get(method, [])
            tok = "${:.2f} \\pm {:.2f}$".format(statistics.mean(accs), statistics.stdev(accs) if len(accs) > 1 else 0.0) if accs else "---"
            gen = "${:.2f} \\pm {:.2f}$".format(statistics.mean(gen_accs), statistics.stdev(gen_accs) if len(gen_accs) > 1 else 0.0) if gen_accs else "---"
            f.write("{} & {} & {} \\\\\n".format(method_display.get(method, method), tok, gen))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print("Wrote", latex_path)
    return runs


if __name__ == "__main__":
    main()
