#!/bin/bash
# Retrain full_ft and lora_diffusion for 10 seeds (42--51). Save checkpoints in
# experiment dir, run validation, then collect training and val accuracy for all 10 seeds.
#
# Usage:
#   bash scripts/rerun_fullft_lora_diffusion.sh
#   OUTPUT_DIR=/path bash scripts/rerun_fullft_lora_diffusion.sh
#
# On SLURM: sbatch slurm/rerun_fullft_lora_diffusion.slurm

set -e

TASKS="sst2"
METHODS="full_ft lora_diffusion"
NUM_SEEDS=10
START_SEED=42
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"
CONFIG="configs/base_config.yaml"

echo "=========================================="
echo "Retrain Full FT + LoRA-Diffusion (10 seeds each)"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Methods: $METHODS"
echo "Seeds: $START_SEED to $((START_SEED + NUM_SEEDS - 1))"
echo ""

# Step 1: Train full_ft and lora_diffusion for all seeds
echo "Step 1: Training full_ft and lora_diffusion..."
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

# Step 2: Evaluate all checkpoints (validation) to get val accuracy per seed
echo "Step 2: Evaluating checkpoints (validation)..."
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

# Step 3: Collect results (training and val accuracy for all 10 seeds)
ALL_METHODS="full_ft lora_diffusion weight_lora adapters bitfit"
echo "Step 3: Collecting results (train + val accuracy, 10 seeds)..."
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
echo "Training and validation accuracy (10 seeds)"
echo "=========================================="
for method in $METHODS; do
    echo ""
    echo "--- $method ---"
    for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
        dir="$OUTPUT_DIR/sst2_${method}_seed${seed}"
        if [ -f "$dir/training_summary.json" ]; then
            train_acc=$(python3 -c "import json; d=json.load(open('$dir/training_summary.json')); print('%.2f' % (d.get('final_training_metrics',{}).get('accuracy',0)*100))" 2>/dev/null || echo "---")
        else
            train_acc="---"
        fi
        if [ -f "$dir/eval_results.json" ]; then
            val_acc=$(python3 -c "import json; d=json.load(open('$dir/eval_results.json')); print('%.2f' % (d.get('metrics',{}).get('accuracy',0)*100))" 2>/dev/null || echo "---")
        else
            val_acc="---"
        fi
        echo "  seed $seed: train_acc=$train_acc%, val_acc=$val_acc%"
    done
done
echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
echo "  - Checkpoints: $OUTPUT_DIR/sst2_full_ft_seed*, sst2_lora_diffusion_seed*"
echo "  - collected_results_with_stats.json, paper_tables_with_stats.tex"
echo ""
