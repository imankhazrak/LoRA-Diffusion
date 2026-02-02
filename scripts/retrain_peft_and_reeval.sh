#!/bin/bash
# Retrain Weight LoRA and Adapters (fixed config), re-eval BitFit, then collect and tables.
# Use after validation-code fixes so val acc for weight_lora/adapters/bitfit is correct.
#
# Prerequisites:
#   - Activate project venv first (e.g. source $HOME/LoRA-Diffusion/venv/bin/activate)
#     or run on SLURM with slurm/retrain_peft_and_reeval.slurm where the venv is activated.
#   - OUTPUT_DIR should already contain full_ft and lora_diffusion runs
#     (sst2_full_ft_seed42..51, sst2_lora_diffusion_seed42..51) for full paper tables.
#
# Usage:
#   source /path/to/venv/bin/activate
#   bash scripts/retrain_peft_and_reeval.sh
#   OUTPUT_DIR=/path/to/outputs bash scripts/retrain_peft_and_reeval.sh
#
# On SLURM (recommended): sbatch slurm/retrain_peft_and_reeval.slurm

set -e

TASKS="sst2"
RETRAIN_METHODS="weight_lora adapters"
ALL_METHODS="full_ft lora_diffusion weight_lora adapters bitfit"
NUM_SEEDS=10
START_SEED=42
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"
CONFIG="configs/base_config.yaml"

echo "=========================================="
echo "Retrain PEFT + Re-eval BitFit + Collect"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Retrain methods: $RETRAIN_METHODS"
echo "Seeds: $START_SEED to $((START_SEED + NUM_SEEDS - 1))"
echo ""

# Step 1: Retrain Weight LoRA and Adapters only
echo "Step 1: Retraining Weight LoRA and Adapters..."
echo "=========================================="
python scripts/run_experiments.py \
  --tasks $TASKS \
  --methods $RETRAIN_METHODS \
  --num_seeds $NUM_SEEDS \
  --start_seed $START_SEED \
  --output_dir "$OUTPUT_DIR" \
  --config "$CONFIG"

if [ $? -ne 0 ]; then
    echo "Error: Retraining failed"
    exit 1
fi
echo ""

# Step 2: Re-run evaluation for BitFit only (existing checkpoints)
echo "Step 2: Re-evaluating BitFit checkpoints (validation only)..."
echo "=========================================="
for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
    checkpoint_dir="$OUTPUT_DIR/sst2_bitfit_seed${seed}"
    if [ -d "$checkpoint_dir" ]; then
        echo "Evaluating: sst2_bitfit_seed${seed}"
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

# Step 3: Run evaluation for newly trained Weight LoRA and Adapters
echo "Step 3: Evaluating new Weight LoRA and Adapters checkpoints..."
echo "=========================================="
for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
    for method in $RETRAIN_METHODS; do
        checkpoint_dir="$OUTPUT_DIR/sst2_${method}_seed${seed}"
        if [ -d "$checkpoint_dir" ]; then
            echo "Evaluating: sst2_${method}_seed${seed}"
            python scripts/evaluate.py \
              --checkpoint "$checkpoint_dir" \
              --task sst2 \
              --split validation \
              --output_file "$checkpoint_dir/eval_results.json" || true
        fi
    done
done
echo ""

# Step 4: Collect results (all 5 methods) and generate paper tables
echo "Step 4: Collecting results and generating paper tables..."
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
echo "  - collected_results_with_stats.json"
echo "  - paper_tables_with_stats.tex"
echo ""
