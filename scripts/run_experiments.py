#!/usr/bin/env python3
"""Run multiple experiments in batch."""

import argparse
import subprocess
import itertools
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch experiments")
    
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="List of tasks to run",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        required=True,
        help="List of methods to run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        help="List of random seeds (default: 42-51 for 10 seeds)",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Number of seeds to generate (starting from --start_seed)",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=42,
        help="Starting seed when using --num_seeds",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/experiments",
        help="Base output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Base config file",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (passed to train.py)",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Max training samples (data subset, passed to train.py)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (use with caution)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    
    return parser.parse_args()


def run_experiment(task, method, seed, output_dir, config, dry_run=False, max_steps=None, subset_size=None):
    """Run a single experiment."""
    experiment_name = f"{task}_{method}_seed{seed}"
    exp_output_dir = Path(output_dir) / experiment_name
    
    cmd = [
        "python",
        "scripts/train.py",
        "--config", config,
        "--task", task,
        "--method", method,
        "--seed", str(seed),
        "--output_dir", str(exp_output_dir),
    ]
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    if subset_size is not None:
        cmd.extend(["--subset_size", str(subset_size)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return 0
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Completed: {experiment_name}")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {experiment_name}")
        err_msg = (e.stderr or "").strip() or (e.stdout or "").strip()
        if err_msg:
            logger.error(f"Error (stderr/stdout): {err_msg[:2000]}")
        else:
            logger.error(f"Exit code: {e.returncode} (no stderr/stdout captured)")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error running {experiment_name}: {e}")
        return 1


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Generate seeds if --num_seeds is specified
    if args.num_seeds is not None:
        args.seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
        logger.info(f"Generated {args.num_seeds} seeds: {args.seeds}")
    
    # Generate all combinations
    experiments = list(itertools.product(args.tasks, args.methods, args.seeds))
    
    logger.info("=" * 80)
    logger.info(f"Running {len(experiments)} experiments")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info("=" * 80)
    
    if args.parallel:
        logger.warning("Parallel execution not implemented. Running sequentially.")
    
    # Run experiments
    failed = []
    successful = []
    start_time = time.time()
    
    for i, (task, method, seed) in enumerate(experiments, 1):
        exp_start = time.time()
        msg = f"\n[{i}/{len(experiments)}] Starting {task}_{method}_seed{seed}"
        logger.info(msg)
        print(msg, flush=True)
        
        exit_code = run_experiment(
            task=task,
            method=method,
            seed=seed,
            output_dir=args.output_dir,
            config=args.config,
            dry_run=args.dry_run,
            max_steps=args.max_steps,
            subset_size=args.subset_size,
        )
        
        exp_elapsed = time.time() - exp_start
        if exit_code != 0:
            failed.append((task, method, seed))
        else:
            successful.append((task, method, seed, exp_elapsed))
        
        # ETA estimation
        if i < len(experiments):
            avg_time_per_exp = (time.time() - start_time) / i
            remaining_exps = len(experiments) - i
            eta_seconds = avg_time_per_exp * remaining_exps
            eta_hours = eta_seconds / 3600
            progress_msg = (f"Progress: {i}/{len(experiments)} ({i/len(experiments)*100:.1f}%) | "
                            f"ETA: {eta_hours:.2f}h | Avg time/exp: {avg_time_per_exp/60:.1f}min")
            logger.info(progress_msg)
            print(progress_msg, flush=True)
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Successful: {len(experiments) - len(failed)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total time: {elapsed_time / 3600:.2f} hours")
    
    if failed:
        logger.info("\nFailed experiments:")
        for task, method, seed in failed:
            logger.info(f"  - {task}_{method}_seed{seed}")
    
    logger.info("\nAll experiments complete!")


if __name__ == "__main__":
    main()
