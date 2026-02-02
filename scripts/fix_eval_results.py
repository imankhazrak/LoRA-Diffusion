#!/usr/bin/env python3
"""Fix evaluation results by recomputing accuracy from saved predictions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_accuracy

def main():
    parser = argparse.ArgumentParser(description="Fix evaluation results")
    parser.add_argument("--job-id", type=str, required=True, help="Job ID")
    parser.add_argument("--task", type=str, default="sst2", help="Task name")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--base-dir", type=str, default="lora-diffusion", help="Base directory")
    
    args = parser.parse_args()
    
    methods = ["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit"]
    base_dir = Path(args.base_dir)
    
    label_names = ["negative", "positive"]
    
    for method in methods:
        checkpoint_dir = base_dir / "checkpoints" / f"{args.task}_{method}_seed{args.seed}_{args.job_id}"
        output_dir = base_dir / "outputs" / f"{args.task}_{method}_seed{args.seed}_{args.job_id}"
        
        # Find latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
        if not checkpoints:
            print(f"⚠ No checkpoint found for {method}, skipping")
            continue
        
        latest_checkpoint = checkpoints[-1]
        predictions_file = latest_checkpoint / "predictions_validation.json"
        eval_output = output_dir / "eval_results.json"
        
        if not predictions_file.exists():
            print(f"⚠ No predictions file for {method}, skipping")
            continue
        
        print(f"\nFixing {method}...")
        
        # Load predictions
        with open(predictions_file, "r") as f:
            pred_data = json.load(f)
        
        predictions = pred_data["predictions"]
        references = pred_data["references"]
        
        # Recompute accuracy with correct decoding
        task_config = {"task": {"type": "classification", "label_names": label_names}}
        accuracy = compute_accuracy(predictions, references, task_config=task_config)
        
        # Update eval_results.json
        if eval_output.exists():
            with open(eval_output, "r") as f:
                eval_data = json.load(f)
        else:
            eval_data = {
                "checkpoint": str(latest_checkpoint),
                "task": args.task,
                "split": "validation",
                "num_samples": len(predictions),
            }
        
        eval_data["metrics"] = {"accuracy": accuracy}
        
        with open(eval_output, "w") as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"✓ {method}: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
