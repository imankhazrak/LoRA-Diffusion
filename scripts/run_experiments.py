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
        default=[42],
        help="List of random seeds",
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


def run_experiment(task, method, seed, output_dir, config, dry_run=False):
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
        logger.error(f"Error: {e.stderr}")
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
    start_time = time.time()
    
    for i, (task, method, seed) in enumerate(experiments, 1):
        logger.info(f"\n[{i}/{len(experiments)}] Starting {task}_{method}_seed{seed}")
        
        exit_code = run_experiment(
            task=task,
            method=method,
            seed=seed,
            output_dir=args.output_dir,
            config=args.config,
            dry_run=args.dry_run,
        )
        
        if exit_code != 0:
            failed.append((task, method, seed))
    
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
