# Verification: Complete Results Saving

This document verifies that all results are saved completely during training and evaluation.

## What Gets Saved During Training

### 1. Training History (`training_history.json`)
**Location:** `{output_dir}/training_history.json`

**Contents:**
- Step-by-step training metrics (loss, accuracy, etc.)
- Learning rate at each logging step
- Time per step
- Saved every `logging_steps` during training

**Format:**
```json
[
  {
    "step": 10,
    "metrics": {
      "loss": 2.3456,
      "ce_loss": 2.3000,
      "reg_loss": 0.0456,
      "accuracy": 0.5234
    },
    "lr": 1e-4,
    "time_per_step": 0.123
  },
  ...
]
```

### 2. Evaluation History (`evaluation_history.json`)
**Location:** `{output_dir}/evaluation_history.json`

**Contents:**
- Evaluation metrics at each evaluation step
- Saved every `eval_frequency` steps
- Updated immediately after each evaluation

**Format:**
```json
[
  {
    "step": 500,
    "metrics": {
      "loss": 1.2345,
      "accuracy": 0.8765,
      "f1": 0.8543
    }
  },
  ...
]
```

### 3. Training Summary (`training_summary.json`)
**Location:** `{output_dir}/training_summary.json`

**Contents:**
- Total training steps
- Best metric achieved
- Final training metrics
- Final evaluation metrics
- Complete configuration used

**Format:**
```json
{
  "total_steps": 5000,
  "best_metric": 0.9234,
  "final_training_metrics": {...},
  "final_eval_metrics": {...},
  "config": {...}
}
```

### 4. Checkpoints
**Location:** `{checkpoint_dir}/{checkpoint_name}/`

**Contents per checkpoint:**
- `config.json` - Full configuration
- `model.pt` - Base model weights (if trainable)
- `lora_module.pt` - LoRA adapter weights
- `router.pt` - Router weights (if multi-task)
- `optimizer.pt` - Optimizer and scheduler state
- `checkpoint_metadata.json` - Checkpoint metadata

**Checkpoint Types:**
- `checkpoint-{step}` - Periodic checkpoints (every `save_frequency` steps)
- `best_model` - Best model based on primary metric
- `final_model` - Final checkpoint after training completes

### 5. Logs
**Location:** `{log_dir}/{experiment_name}.log`

**Contents:**
- All training logs
- Error messages
- Progress information
- Configuration details

## What Gets Saved During Evaluation

### 1. Evaluation Results (`test_results.json` or custom path)
**Location:** Specified by `--output_file` or `{checkpoint_dir}/test_results.json`

**Contents:**
- Checkpoint path
- Task and split information
- Number of samples evaluated
- All computed metrics

**Format:**
```json
{
  "checkpoint": "/path/to/checkpoint",
  "task": "sst2",
  "split": "test",
  "num_samples": 1821,
  "metrics": {
    "accuracy": 0.9234,
    "f1": 0.9102
  }
}
```

### 2. Predictions (`predictions_{split}.json`)
**Location:** `{checkpoint_dir}/predictions_{split}.json`

**Contents:**
- All predictions
- All reference labels/texts

**Format:**
```json
{
  "predictions": ["positive", "negative", ...],
  "references": ["positive", "negative", ...]
}
```

## Verification Checklist

Before running jobs, verify:

- [ ] **Training History Saved**: Check `training_history.json` exists and has entries
- [ ] **Evaluation History Saved**: Check `evaluation_history.json` exists and has entries
- [ ] **Summary Saved**: Check `training_summary.json` exists
- [ ] **Checkpoints Created**: Verify checkpoints directory has `best_model`, `final_model`, and periodic checkpoints
- [ ] **Logs Written**: Check log file exists and contains training information
- [ ] **Evaluation Results**: If running evaluation, verify results JSON and predictions JSON exist

## How to Verify After Job Completion

### 1. Check Output Directory Structure

```bash
# List output directory
ls -lh /fs/scratch/users/$USER/lora-diffusion/outputs/<task>_<method>_<job_id>/

# Should see:
# - training_history.json
# - evaluation_history.json
# - training_summary.json
```

### 2. Verify Training History

```bash
# Check training history exists and has data
cat outputs/<task>_<method>_<job_id>/training_history.json | jq 'length'
# Should return number of logging steps

# Check last entry
cat outputs/<task>_<method>_<job_id>/training_history.json | jq '.[-1]'
```

### 3. Verify Evaluation History

```bash
# Check evaluation history exists
cat outputs/<task>_<method>_<job_id>/evaluation_history.json | jq 'length'
# Should return number of evaluations

# Check best metric
cat outputs/<task>_<method>_<job_id>/evaluation_history.json | jq '[.[] | .metrics.accuracy] | max'
```

### 4. Verify Checkpoints

```bash
# List checkpoints
ls -lh /fs/scratch/users/$USER/lora-diffusion/checkpoints/<task>_<method>_<job_id>/

# Should see:
# - best_model/
# - final_model/
# - checkpoint-1000/
# - checkpoint-2000/
# etc.

# Check checkpoint contents
ls -lh checkpoints/<task>_<method>_<job_id>/best_model/
# Should see: config.json, lora_module.pt, optimizer.pt, checkpoint_metadata.json
```

### 5. Verify Summary

```bash
# View training summary
cat outputs/<task>_<method>_<job_id>/training_summary.json | jq '.'

# Should show:
# - total_steps
# - best_metric
# - final metrics
# - config
```

### 6. Verify Logs

```bash
# Check log file exists
ls -lh logs/<experiment_name>.log

# Check log has content
tail -n 100 logs/<experiment_name>.log

# Check for errors
grep -i error logs/<experiment_name>.log
```

## Complete File Structure After Training

```
outputs/
└── <task>_<method>_<job_id>/
    ├── training_history.json          # Training metrics over time
    ├── evaluation_history.json        # Evaluation metrics over time
    └── training_summary.json           # Final summary

checkpoints/
└── <task>_<method>_<job_id>/
    ├── best_model/
    │   ├── config.json
    │   ├── lora_module.pt
    │   ├── optimizer.pt
    │   └── checkpoint_metadata.json
    ├── final_model/
    │   ├── config.json
    │   ├── lora_module.pt
    │   ├── optimizer.pt
    │   └── checkpoint_metadata.json
    └── checkpoint-{step}/
        ├── config.json
        ├── lora_module.pt
        ├── optimizer.pt
        └── checkpoint_metadata.json

logs/
└── <experiment_name>.log              # Training logs
```

## After Evaluation

```
checkpoints/
└── <task>_<method>_<job_id>/
    └── <checkpoint_name>/
        ├── predictions_test.json      # Predictions and references
        └── (if --output_file specified)
            └── test_results.json      # Evaluation results
```

## Quick Verification Script

Save this as `verify_results.sh`:

```bash
#!/bin/bash
# Verify all results are saved

OUTPUT_DIR=$1
CHECKPOINT_DIR=$2

echo "Checking output directory: $OUTPUT_DIR"
[ -f "$OUTPUT_DIR/training_history.json" ] && echo "✓ training_history.json exists" || echo "✗ training_history.json MISSING"
[ -f "$OUTPUT_DIR/evaluation_history.json" ] && echo "✓ evaluation_history.json exists" || echo "✗ evaluation_history.json MISSING"
[ -f "$OUTPUT_DIR/training_summary.json" ] && echo "✓ training_summary.json exists" || echo "✗ training_summary.json MISSING"

echo ""
echo "Checking checkpoint directory: $CHECKPOINT_DIR"
[ -d "$CHECKPOINT_DIR/best_model" ] && echo "✓ best_model checkpoint exists" || echo "✗ best_model MISSING"
[ -d "$CHECKPOINT_DIR/final_model" ] && echo "✓ final_model checkpoint exists" || echo "✗ final_model MISSING"

echo ""
echo "Checking checkpoint contents:"
[ -f "$CHECKPOINT_DIR/best_model/config.json" ] && echo "✓ best_model/config.json exists" || echo "✗ best_model/config.json MISSING"
[ -f "$CHECKPOINT_DIR/best_model/lora_module.pt" ] && echo "✓ best_model/lora_module.pt exists" || echo "✗ best_model/lora_module.pt MISSING"
[ -f "$CHECKPOINT_DIR/best_model/optimizer.pt" ] && echo "✓ best_model/optimizer.pt exists" || echo "✗ best_model/optimizer.pt MISSING"
[ -f "$CHECKPOINT_DIR/best_model/checkpoint_metadata.json" ] && echo "✓ best_model/checkpoint_metadata.json exists" || echo "✗ best_model/checkpoint_metadata.json MISSING"
```

Usage:
```bash
chmod +x verify_results.sh
./verify_results.sh \
  /fs/scratch/users/$USER/lora-diffusion/outputs/sst2_lora_diffusion_12345 \
  /fs/scratch/users/$USER/lora-diffusion/checkpoints/sst2_lora_diffusion_12345
```

## Summary

✅ **All results are now saved completely:**
- Training metrics (every logging step)
- Evaluation metrics (every evaluation step)
- Training summary (final)
- Checkpoints (periodic, best, final)
- Logs (all training information)
- Evaluation results (if running evaluation)
- Predictions (if running evaluation)

The code has been enhanced to ensure nothing is lost!
