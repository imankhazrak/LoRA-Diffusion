#!/bin/bash
# Re-run validation evaluation for ALL 5 models (full_ft, lora_diffusion, weight_lora,
# adapters, bitfit) with 10 seeds (42-51), same as training seeds. Evaluation only, no training.
#
# Usage:
#   bash scripts/reevaluate_all_models_10seeds.sh
#   OUTPUT_DIR=/path bash scripts/reevaluate_all_models_10seeds.sh
#
# On SLURM: sbatch slurm/reevaluate_all_models_10seeds.slurm

set -e

ALL_METHODS="full_ft lora_diffusion weight_lora adapters bitfit"
TASK="sst2"
START_SEED=42
NUM_SEEDS=10
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"

echo "=========================================="
echo "Re-evaluate val accuracy: all 5 models, seeds 42-51"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Methods: $ALL_METHODS"
echo "Seeds: $START_SEED to $((START_SEED + NUM_SEEDS - 1))"
echo ""

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

# Step 1: Re-run validation evaluation for each checkpoint (50 runs total)
echo "Step 1: Re-evaluating checkpoints (validation, generation-based)..."
echo "=========================================="
for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
    for method in $ALL_METHODS; do
        checkpoint_dir="$OUTPUT_DIR/${TASK}_${method}_seed${seed}"
        if [ -d "$checkpoint_dir" ]; then
            if [ -f "$checkpoint_dir/training_summary.json" ]; then
                echo "Evaluating: ${TASK}_${method}_seed${seed}"
                python scripts/evaluate.py \
                    --checkpoint "$checkpoint_dir" \
                    --task "$TASK" \
                    --split validation \
                    --output_file "$checkpoint_dir/eval_results.json" \
                    --device cuda || { echo "  Warning: eval failed for $checkpoint_dir"; }
            else
                echo "Skipping (no training_summary.json): $checkpoint_dir"
            fi
        else
            echo "Skipping (not found): $checkpoint_dir"
        fi
    done
done
echo ""

# Step 2: Collect results for all 5 methods and generate paper tables
echo "Step 2: Collecting results and generating paper tables..."
echo "=========================================="
python scripts/collect_results_with_stats.py \
    --base_dir "$OUTPUT_DIR" \
    --task "$TASK" \
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
echo "  - collected_results_with_stats.json"
echo "  - paper_tables_with_stats.tex"
echo ""
