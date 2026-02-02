#!/bin/bash
# Re-evaluate validation accuracy for full_ft, lora_diffusion, and bitfit only.
# Use after evaluation-code fixes to get correct val accuracy on existing checkpoints.
# If these results look correct, proceed to retrain weight_lora and adapters.
#
# Usage:
#   bash scripts/reevaluate_fullft_lora_bitfit.sh
#   OUTPUT_DIR=/path/to/outputs bash scripts/reevaluate_fullft_lora_bitfit.sh
#
# On SLURM: sbatch slurm/reevaluate_fullft_lora_bitfit.slurm

set -e

REEVAL_METHODS="full_ft lora_diffusion bitfit"
TASK="sst2"
START_SEED=42
NUM_SEEDS=10
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"

echo "=========================================="
echo "Re-evaluate: full_ft, LoRA-Diffusion, BitFit (val accuracy)"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Methods: $REEVAL_METHODS"
echo "Seeds: $START_SEED to $((START_SEED + NUM_SEEDS - 1))"
echo ""

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

# Step 1: Re-run validation evaluation for each checkpoint
echo "Step 1: Re-evaluating checkpoints (validation, generation-based)..."
echo "=========================================="
for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
    for method in $REEVAL_METHODS; do
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

# Step 2: Collect results for the 3 methods and generate summary table
echo "Step 2: Collecting results and generating summary..."
echo "=========================================="
python scripts/collect_results_with_stats.py \
    --base_dir "$OUTPUT_DIR" \
    --task "$TASK" \
    --methods $REEVAL_METHODS \
    --seeds "$START_SEED-$((START_SEED + NUM_SEEDS - 1))" \
    --output collected_results_reeval_3methods.json \
    --baseline full_ft

python scripts/generate_paper_tables.py \
    --results collected_results_reeval_3methods.json \
    --output paper_tables_reeval_3methods.tex

echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
echo "  - Check collected_results_reeval_3methods.json and paper_tables_reeval_3methods.tex"
echo "  - If results look correct, run retrain for weight_lora and adapters:"
echo "      bash scripts/retrain_peft_and_reeval.sh"
echo "    or: sbatch slurm/retrain_peft_and_reeval.slurm"
echo ""
