#!/usr/bin/env python3
"""
Run multi-task composition evaluations and fill Table 12.

For each of SST-2, MRPC, QNLI, runs evaluation under four settings:
- Single-task: adapter only (no composition)
- Composed (router): composer with trained router
- Average: composer with uniform weights (1/M)
- Task arithmetic: composer with sum of deltas

Saves results to multitask_composition_results.json and optionally updates
doc/Paper.tex Table 12 (tab:multitask).
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Rows and column order for Table 12
TASKS = ["sst2", "mrpc", "qnli"]
COLUMNS = ["single_task", "composed_router", "average", "task_arithmetic"]


def run_evaluate(
    checkpoint: str,
    task: str,
    use_composition: bool = False,
    task_modules: list = None,
    task_names: list = None,
    composition_mode: str = "router",
    router_path: str = None,
    output_file: str = None,
    config: str = None,
) -> dict:
    """Run scripts/evaluate.py and return parsed results. (evaluate.py loads config from checkpoint.)"""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate.py"),
        "--checkpoint", checkpoint,
        "--task", task,
        "--split", "validation",
        "--eval_classification_head",
    ]
    if output_file:
        cmd += ["--output_file", output_file]
    if use_composition:
        cmd += ["--use_composition", "--task_modules"] + task_modules + ["--task_names"] + task_names
        cmd += ["--composition_mode", composition_mode]
        if router_path:
            cmd += ["--router_path", router_path]
    root = Path(__file__).parent.parent
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=root)
        if out.returncode != 0:
            logger.warning(f"evaluate.py failed: {out.stderr[:500]}")
            return {}
        if output_file and Path(output_file).exists():
            with open(output_file) as f:
                data = json.load(f)
            return data.get("metrics", {})
    except Exception as e:
        logger.warning(f"Run failed: {e}")
    return {}


def main():
    ap = argparse.ArgumentParser(description="Run multi-task table evaluations")
    ap.add_argument("--sst2_ckpt", type=str, required=True, help="Path to SST-2 checkpoint dir")
    ap.add_argument("--mrpc_ckpt", type=str, required=True, help="Path to MRPC checkpoint dir")
    ap.add_argument("--qnli_ckpt", type=str, required=True, help="Path to QNLI checkpoint dir")
    ap.add_argument("--router_path", type=str, default=None, help="Path to router.pt (for composed router column)")
    ap.add_argument("--output", type=str, default="multitask_composition_results.json")
    ap.add_argument("--update_paper", action="store_true", help="Update doc/Paper.tex Table 12")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(__file__).parent.parent
    checkpoints = {"sst2": args.sst2_ckpt, "mrpc": args.mrpc_ckpt, "qnli": args.qnli_ckpt}
    task_modules = [checkpoints[t] for t in TASKS]
    task_names = list(TASKS)
    results = {task: {} for task in TASKS}

    # Single-task column
    for task in TASKS:
        out_file = str(root / f"_mt_{task}_single.json")
        m = run_evaluate(
            checkpoint=checkpoints[task],
            task=task,
            output_file=out_file,
            config=args.config,
        )
        acc = m.get("classification_head_val_acc")
        results[task]["single_task"] = round(acc, 2) if acc is not None else None
        if Path(out_file).exists():
            Path(out_file).unlink()
        logger.info(f"{task} single-task: {results[task]['single_task']}")

    # Composed (router)
    for task in TASKS:
        out_file = str(root / f"_mt_{task}_router.json")
        m = run_evaluate(
            checkpoint=checkpoints["sst2"],
            task=task,
            use_composition=True,
            task_modules=task_modules,
            task_names=task_names,
            composition_mode="router",
            router_path=args.router_path,
            output_file=out_file,
        )
        acc = m.get("classification_head_val_acc")
        results[task]["composed_router"] = round(acc, 2) if acc is not None else None
        if Path(out_file).exists():
            Path(out_file).unlink()
        logger.info(f"{task} composed (router): {results[task]['composed_router']}")

    # Average
    for task in TASKS:
        out_file = str(root / f"_mt_{task}_avg.json")
        m = run_evaluate(
            checkpoint=checkpoints["sst2"],
            task=task,
            use_composition=True,
            task_modules=task_modules,
            task_names=task_names,
            composition_mode="uniform",
            output_file=out_file,
        )
        acc = m.get("classification_head_val_acc")
        results[task]["average"] = round(acc, 2) if acc is not None else None
        if Path(out_file).exists():
            Path(out_file).unlink()
        logger.info(f"{task} average: {results[task]['average']}")

    # Task arithmetic
    for task in TASKS:
        out_file = str(root / f"_mt_{task}_arith.json")
        m = run_evaluate(
            checkpoint=checkpoints["sst2"],
            task=task,
            use_composition=True,
            task_modules=task_modules,
            task_names=task_names,
            composition_mode="task_arithmetic",
            output_file=out_file,
        )
        acc = m.get("classification_head_val_acc")
        results[task]["task_arithmetic"] = round(acc, 2) if acc is not None else None
        if Path(out_file).exists():
            Path(out_file).unlink()
        logger.info(f"{task} task_arithmetic: {results[task]['task_arithmetic']}")

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out_path}")

    if args.update_paper:
        paper = root / "doc" / "Paper.tex"
        if not paper.exists():
            logger.warning(f"Paper not found: {paper}")
            return 0
        text = paper.read_text()
        def cell(v):
            return f"{v:.2f}" if v is not None else "(to be filled)"
        # Replace table body
        old = (
            "SST-2 & (to be filled) & (to be filled) & (to be filled) & (to be filled) \\\\\n"
            "MRPC & (to be filled) & (to be filled) & (to be filled) & (to be filled) \\\\\n"
            "QNLI & (to be filled) & (to be filled) & (to be filled) & (to be filled) \\\\"
        )
        new = (
            f"SST-2 & {cell(results['sst2']['single_task'])} & {cell(results['sst2']['composed_router'])} & {cell(results['sst2']['average'])} & {cell(results['sst2']['task_arithmetic'])} \\\\\n"
            f"MRPC & {cell(results['mrpc']['single_task'])} & {cell(results['mrpc']['composed_router'])} & {cell(results['mrpc']['average'])} & {cell(results['mrpc']['task_arithmetic'])} \\\\\n"
            f"QNLI & {cell(results['qnli']['single_task'])} & {cell(results['qnli']['composed_router'])} & {cell(results['qnli']['average'])} & {cell(results['qnli']['task_arithmetic'])} \\\\"
        )
        if old in text:
            text = text.replace(old, new)
            paper.write_text(text)
            logger.info("Updated doc/Paper.tex Table 12")
        else:
            logger.warning("Table 12 placeholder not found in Paper.tex; not updated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
