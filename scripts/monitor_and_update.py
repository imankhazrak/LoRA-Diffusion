#!/usr/bin/env python3
"""Monitor job completion and automatically collect results and update paper tables."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def check_job_status(job_id: str) -> tuple[bool, str]:
    """Check if job is still running."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "--format=%T"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            status = result.stdout.strip()
            return status in ["R", "PD", "CF"], status
        else:
            # Job not in queue - might be completed
            return False, "COMPLETED"
    except Exception as e:
        return None, str(e)


def wait_for_job_completion(job_id: str, check_interval: int = 300) -> bool:
    """Wait for job to complete, checking every check_interval seconds."""
    print(f"Monitoring job {job_id}...")
    
    while True:
        is_running, status = check_job_status(job_id)
        
        if is_running is None:
            print(f"Error checking job status: {status}")
            return False
        
        if not is_running:
            print(f"Job {job_id} completed (status: {status})")
            return True
        
        print(f"Job {job_id} still running (status: {status}). Waiting {check_interval}s...")
        time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor job and update paper tables")
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="SLURM job ID",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sst2",
        help="Task name",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job completion (otherwise just check once)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="Seconds between checks when waiting",
    )
    parser.add_argument(
        "--parameter-counts",
        type=str,
        default="parameter_counts.json",
        help="Path to parameter_counts.json",
    )
    parser.add_argument(
        "--paper-tex",
        type=str,
        default="doc/Paper.tex",
        help="Path to Paper.tex",
    )
    
    args = parser.parse_args()
    
    # Check job status
    is_running, status = check_job_status(args.job_id)
    
    if is_running:
        print(f"Job {args.job_id} is still running (status: {status})")
        if args.wait:
            if not wait_for_job_completion(args.job_id, args.check_interval):
                sys.exit(1)
        else:
            print("Use --wait to wait for completion")
            sys.exit(0)
    else:
        print(f"Job {args.job_id} appears to be completed (status: {status})")
    
    # Collect results
    print("\nCollecting results...")
    collect_cmd = [
        "python", "scripts/collect_results.py",
        "--parameter-counts", args.parameter_counts,
        "--task", args.task,
        "--job-id", args.job_id,
        "--output", "collected_results.json",
    ]
    
    result = subprocess.run(collect_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error collecting results: {result.stderr}")
        sys.exit(1)
    
    print("Results collected successfully")
    
    # Update paper tables
    print("\nUpdating paper tables...")
    update_cmd = [
        "python", "scripts/update_paper_tables.py",
        "--collected-results", "collected_results.json",
        "--paper-tex", args.paper_tex,
        "--handle-discrepancy",
    ]
    
    result = subprocess.run(update_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error updating tables: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    print("\nPaper tables updated successfully!")


if __name__ == "__main__":
    main()
