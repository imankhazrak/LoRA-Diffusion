#!/usr/bin/env python3
"""Re-evaluate all models with corrected accuracy computation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate all models")
    parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    parser.add_argument("--task", type=str, default="sst2", help="Task name")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--base-dir", type=str, default="lora-diffusion", help="Base directory")
    
    args = parser.parse_args()
    
    methods = ["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit"]
    base_dir = Path(args.base_dir)
    
    for method in methods:
        checkpoint_dir = base_dir / "checkpoints" / f"{args.task}_{method}_seed{args.seed}_{args.job_id}"
        output_dir = base_dir / "outputs" / f"{args.task}_{method}_seed{args.seed}_{args.job_id}"
        
        # Find latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
        if not checkpoints:
            print(f"⚠ No checkpoint found for {method}, skipping")
            continue
        
        latest_checkpoint = checkpoints[-1]
        eval_output = output_dir / "eval_results.json"
        
        print(f"\nEvaluating {method} from {latest_checkpoint}")
        
        cmd = [
            "python", "scripts/evaluate.py",
            "--checkpoint", str(latest_checkpoint),
            "--task", args.task,
            "--split", "validation",
            "--output_file", str(eval_output),
            "--device", "cuda",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and eval_output.exists():
            with open(eval_output, "r") as f:
                eval_data = json.load(f)
            acc = eval_data.get("metrics", {}).get("accuracy", 0)
            print(f"✓ {method}: {acc:.4f} ({acc*100:.2f}%)")
        else:
            print(f"✗ {method} evaluation failed")
            print(result.stderr)

if __name__ == "__main__":
    main()
