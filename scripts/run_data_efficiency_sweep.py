#!/usr/bin/env python3
"""
Run data-efficiency sweep: train LoRA-Diffusion and Weight LoRA on SST-2 at
different training-set fractions, collect validation accuracy, and save
results for the data efficiency figure (Figure 3).

Output: data_efficiency_results.json (used by scripts/generate_figures.py
generate_data_efficiency()).

Usage:
  python scripts/run_data_efficiency_sweep.py --output_dir ./outputs/data_efficiency_sweep
  python scripts/run_data_efficiency_sweep.py --output_dir ./outputs/data_efficiency_sweep --seeds 42 43 44
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# SST-2 (SetFit/sst2) train split size
SST2_TRAIN_SIZE = 67349

DEFAULT_FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
METHODS = ["lora_diffusion", "weight_lora"]


def parse_args():
    p = argparse.ArgumentParser(description="Run data-efficiency sweep for Figure 3")
    p.add_argument("--output_dir", type=str, default="./outputs/data_efficiency_sweep",
                   help="Base output directory for sweep runs")
    p.add_argument("--config", type=str, default="configs/base_config.yaml")
    p.add_argument("--task", type=str, default="sst2")
    p.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS,
                   help="Training data fractions (e.g. 0.1 0.2 0.4 0.6 0.8 1.0)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Seeds per (method, fraction) for mean/std")
    p.add_argument("--results_file", type=str, default="data_efficiency_results.json",
                   help="Output JSON path (saved in project root or --output_dir)")
    p.add_argument("--dry_run", action="store_true", help="Print commands only")
    return p.parse_args()


def run_train(task, method, seed, output_dir, config, subset_size, max_steps, dry_run):
    """Run train.py; return 0 on success."""
    exp_name = f"{task}_{method}_seed{seed}_pct{int(subset_size * 100) if subset_size < 1.0 else 100}"
    exp_dir = Path(output_dir) / exp_name
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", config,
        "--task", task,
        "--method", method,
        "--seed", str(seed),
        "--output_dir", str(exp_dir),
    ]
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    if subset_size < 1.0:
        n_samples = int(SST2_TRAIN_SIZE * subset_size)
        cmd.extend(["--subset_size", str(n_samples)])
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return 0
    ret = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return ret.returncode


def run_evaluate(checkpoint_dir, task, output_file, dry_run):
    """Run evaluate.py; return val accuracy or None."""
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", str(checkpoint_dir),
        "--task", task,
        "--split", "validation",
        "--output_file", str(output_file),
        "--device", "cuda",
    ]
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return None
    ret = subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True)
    if ret.returncode != 0:
        return None
    if not Path(output_file).exists():
        return None
    with open(output_file) as f:
        data = json.load(f)
    return data.get("metrics", {}).get("accuracy")


def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "task": args.task,
        "train_size_full": SST2_TRAIN_SIZE,
        "fractions": args.fractions,
        "lora_diffusion": {},
        "weight_lora": {},
    }

    for fraction in args.fractions:
        frac_key = f"{fraction:.2f}" if fraction < 1.0 else "1.00"
        for method in METHODS:
            values = []
            for seed in args.seeds:
                pct = int(fraction * 100) if fraction < 1.0 else 100
                exp_name = f"{args.task}_{method}_seed{seed}_pct{pct}"
                exp_dir = output_dir / exp_name
                if run_train(
                    args.task, method, seed, str(output_dir), args.config,
                    fraction, max_steps=10000, dry_run=args.dry_run
                ) != 0 and not args.dry_run:
                    print(f"Train failed: {exp_name}", file=sys.stderr)
                    continue
                eval_file = exp_dir / "eval_results.json"
                acc = run_evaluate(exp_dir, args.task, eval_file, args.dry_run)
                if acc is not None:
                    values.append(round(acc * 100, 2))
            if not values:
                results[method][frac_key] = {"mean": None, "std": None, "values": [], "n": 0}
                continue
            mean = round(statistics.mean(values), 2)
            std = round(statistics.stdev(values), 2) if len(values) > 1 else 0.0
            results[method][frac_key] = {"mean": mean, "std": std, "values": values, "n": len(values)}

    # Save results (project root or next to output_dir)
    results_path = Path(args.results_file)
    if not results_path.is_absolute():
        results_path = project_root / results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
