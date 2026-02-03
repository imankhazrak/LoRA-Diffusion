#!/bin/bash
# Re-run weight_lora and adapters training only (10 seeds each, full steps) so checkpoints
# (model.pt) are saved in the experiment dir. Then re-evaluate. Same approach as BitFit rerun.
#
# Usage:
#   bash scripts/rerun_weightlora_adapters.sh
#   OUTPUT_DIR=/path bash scripts/rerun_weightlora_adapters.sh
#
# On SLURM: sbatch slurm/rerun_weightlora_adapters.slurm

set -e

TASKS="sst2"
METHODS="weight_lora adapters"
NUM_SEEDS=10
START_SEED=42
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"
CONFIG="configs/base_config.yaml"

echo "=========================================="
echo "Re-run Weight LoRA + Adapters (10 seeds each, full steps)"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Methods: $METHODS"
echo "Seeds: $START_SEED to $((START_SEED + NUM_SEEDS - 1))"
echo ""

# Step 1: Train weight_lora and adapters for all seeds
echo "Step 1: Training weight_lora and adapters..."
echo "=========================================="
python scripts/run_experiments.py \
  --tasks $TASKS \
  --methods $METHODS \
  --num_seeds $NUM_SEEDS \
  --start_seed $START_SEED \
  --output_dir "$OUTPUT_DIR" \
  --config "$CONFIG"

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi
echo ""

# Step 2: Evaluate all checkpoints (validation)
echo "Step 2: Evaluating checkpoints..."
echo "=========================================="
for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
    for method in $METHODS; do
        checkpoint_dir="$OUTPUT_DIR/sst2_${method}_seed${seed}"
        if [ -d "$checkpoint_dir" ]; then
            echo "Evaluating: sst2_${method}_seed${seed}"
            python scripts/evaluate.py \
              --checkpoint "$checkpoint_dir" \
              --task sst2 \
              --split validation \
              --output_file "$checkpoint_dir/eval_results.json" \
              --device cuda || true
        fi
    done
done
echo ""

# Step 3: Collect results for all 5 methods and generate paper tables
ALL_METHODS="full_ft lora_diffusion weight_lora adapters bitfit"
echo "Step 3: Collecting results and generating paper tables (all 5 methods)..."
echo "=========================================="
python scripts/collect_results_with_stats.py \
  --base_dir "$OUTPUT_DIR" \
  --task sst2 \
  --methods $ALL_METHODS \
  --seeds "$START_SEED-$((START_SEED + NUM_SEEDS - 1))" \
  --output collected_results_with_stats.json \
  --baseline full_ft

python scripts/generate_paper_tables.py \
  --results collected_results_with_stats.json \
  --output paper_tables_with_stats.tex

echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
echo "  - Checkpoints: $OUTPUT_DIR/sst2_weight_lora_seed*, sst2_adapters_seed*"
echo "  - collected_results_with_stats.json, paper_tables_with_stats.tex (all 5 methods)"
echo ""
