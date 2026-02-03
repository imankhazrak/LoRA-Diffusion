#!/bin/bash
# Re-evaluate validation accuracy for weight_lora and adapters only, one seed (42).
# Use to verify val accuracy for these two methods without full retrain.
#
# Usage:
#   bash scripts/reevaluate_weightlora_adapters_one_seed.sh
#   OUTPUT_DIR=/path bash scripts/reevaluate_weightlora_adapters_one_seed.sh
#
# On SLURM: sbatch slurm/reevaluate_weightlora_adapters_one_seed.slurm

set -e

METHODS="weight_lora adapters"
TASK="sst2"
SEED=42
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"

echo "=========================================="
echo "Re-evaluate: weight_lora, adapters (seed $SEED only)"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Methods: $METHODS"
echo "Seed: $SEED"
echo ""

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

echo "Evaluating checkpoints (validation, generation-based)..."
echo "=========================================="
for method in $METHODS; do
    checkpoint_dir="$OUTPUT_DIR/${TASK}_${method}_seed${SEED}"
    if [ -d "$checkpoint_dir" ]; then
        echo "Evaluating: ${TASK}_${method}_seed${SEED}"
        python scripts/evaluate.py \
            --checkpoint "$checkpoint_dir" \
            --task "$TASK" \
            --split validation \
            --output_file "$checkpoint_dir/eval_results.json" \
            --device cuda || { echo "  Warning: eval failed for $checkpoint_dir"; }
    else
        echo "Skipping (not found): $checkpoint_dir"
    fi
done
echo ""

echo "=========================================="
echo "Done"
echo "=========================================="
echo "  - Check eval_results.json in each checkpoint dir for val accuracy"
echo ""
