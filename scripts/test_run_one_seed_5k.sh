#!/bin/bash
# Test run: one seed (42), max_steps=5000, Weight LoRA + Adapters only.
# Use to verify code works before full retrain.
#
# Usage:
#   bash scripts/test_run_one_seed_5k.sh
#   OUTPUT_DIR=/path bash scripts/test_run_one_seed_5k.sh
#
# On SLURM: sbatch slurm/test_run_one_seed_5k.slurm

set -e

TASKS="sst2"
METHODS="weight_lora adapters"
SEED=42
MAX_STEPS=5000
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/test_one_seed_5k}"
CONFIG="configs/base_config.yaml"

echo "=========================================="
echo "Test run: 1 seed, 5000 steps (Weight LoRA + Adapters)"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Methods: $METHODS"
echo "Seed: $SEED"
echo "Max steps: $MAX_STEPS"
echo ""

# Step 1: Train
echo "Step 1: Training..."
echo "=========================================="
python scripts/run_experiments.py \
  --tasks $TASKS \
  --methods $METHODS \
  --seeds $SEED \
  --max_steps $MAX_STEPS \
  --output_dir "$OUTPUT_DIR" \
  --config "$CONFIG"

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi
echo ""

# Step 2: Evaluate
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
          --output_file "$checkpoint_dir/eval_results.json" || true
    else
        echo "Warning: Checkpoint not found: $checkpoint_dir"
    fi
done
echo ""

# Step 3: Collect results and generate tables (test subset)
echo "Step 3: Collecting results and generating tables..."
echo "=========================================="
python scripts/collect_results_with_stats.py \
  --base_dir "$OUTPUT_DIR" \
  --task sst2 \
  --methods $METHODS \
  --seeds "$SEED" \
  --output collected_results_test_5k.json \
  --baseline full_ft

python scripts/generate_paper_tables.py \
  --results collected_results_test_5k.json \
  --output paper_tables_test_5k.tex

echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
echo "  - Checkpoints: $OUTPUT_DIR"
echo "  - collected_results_test_5k.json"
echo "  - paper_tables_test_5k.tex"
echo ""
