#!/bin/bash
# Re-run weight_lora and adapters for ONE seed (42) only, then evaluate.
# Use to verify training + eval work before full 10-seed rerun.
#
# Usage:
#   bash scripts/rerun_weightlora_adapters_one_seed.sh
#   OUTPUT_DIR=/path bash scripts/rerun_weightlora_adapters_one_seed.sh
#
# On SLURM: sbatch slurm/rerun_weightlora_adapters_one_seed.slurm

set -e

TASKS="sst2"
METHODS="weight_lora adapters"
SEED=42
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"
CONFIG="configs/base_config.yaml"

echo "=========================================="
echo "Re-run Weight LoRA + Adapters (1 seed: $SEED)"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Methods: $METHODS"
echo "Seed: $SEED"
echo ""

# Step 1: Train weight_lora and adapters for seed 42 only
echo "Step 1: Training weight_lora and adapters (seed $SEED)..."
echo "=========================================="
python scripts/run_experiments.py \
  --tasks $TASKS \
  --methods $METHODS \
  --seeds $SEED \
  --output_dir "$OUTPUT_DIR" \
  --config "$CONFIG"

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi
echo ""

# Step 2: Evaluate both checkpoints (validation)
echo "Step 2: Evaluating checkpoints..."
echo "=========================================="
for method in $METHODS; do
    checkpoint_dir="$OUTPUT_DIR/sst2_${method}_seed${SEED}"
    if [ -d "$checkpoint_dir" ]; then
        echo "Evaluating: sst2_${method}_seed${SEED}"
        python scripts/evaluate.py \
          --checkpoint "$checkpoint_dir" \
          --task sst2 \
          --split validation \
          --output_file "$checkpoint_dir/eval_results.json" \
          --device cuda || true
    fi
done
echo ""

echo "=========================================="
echo "Done"
echo "=========================================="
echo "  - Checkpoints: $OUTPUT_DIR/sst2_weight_lora_seed${SEED}, sst2_adapters_seed${SEED}"
echo "  - Val accuracy in: .../eval_results.json (metrics.accuracy)"
echo ""
