#!/bin/bash
# Re-run BitFit training only (10 seeds, full steps) so checkpoints (model.pt) are saved
# in the experiment dir. Then re-evaluate and optionally collect.
#
# Usage:
#   bash scripts/rerun_bitfit.sh
#   OUTPUT_DIR=/path bash scripts/rerun_bitfit.sh
#
# On SLURM: sbatch slurm/rerun_bitfit.slurm

set -e

TASKS="sst2"
METHOD="bitfit"
NUM_SEEDS=10
START_SEED=42
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"
CONFIG="configs/base_config.yaml"

echo "=========================================="
echo "Re-run BitFit only (10 seeds, full steps)"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Seeds: $START_SEED to $((START_SEED + NUM_SEEDS - 1))"
echo ""

# Step 1: Train BitFit for all seeds (checkpoints will be in $OUTPUT_DIR/sst2_bitfit_seedXX/)
echo "Step 1: Training BitFit..."
echo "=========================================="
python scripts/run_experiments.py \
  --tasks $TASKS \
  --methods $METHOD \
  --num_seeds $NUM_SEEDS \
  --start_seed $START_SEED \
  --output_dir "$OUTPUT_DIR" \
  --config "$CONFIG"

if [ $? -ne 0 ]; then
    echo "Error: BitFit training failed"
    exit 1
fi
echo ""

# Step 2: Evaluate all BitFit checkpoints (validation)
echo "Step 2: Evaluating BitFit checkpoints..."
echo "=========================================="
for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
    checkpoint_dir="$OUTPUT_DIR/sst2_bitfit_seed${seed}"
    if [ -d "$checkpoint_dir" ]; then
        echo "Evaluating: sst2_bitfit_seed${seed}"
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
echo "  - Checkpoints: $OUTPUT_DIR/sst2_bitfit_seed*"
echo "  - Re-run collect + tables for all 5 methods if needed:"
echo "      python scripts/collect_results_with_stats.py --base_dir $OUTPUT_DIR --task sst2 --seeds 42-51 --output collected_results_with_stats.json"
echo "      python scripts/generate_paper_tables.py --results collected_results_with_stats.json --output paper_tables_with_stats.tex"
echo ""
