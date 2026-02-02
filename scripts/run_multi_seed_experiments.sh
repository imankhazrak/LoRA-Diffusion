#!/bin/bash
# Run multi-seed experiments for statistical analysis

set -e

echo "=========================================="
echo "Multi-Seed Experiment Pipeline"
echo "=========================================="
echo ""

# Configuration
TASKS="sst2"
METHODS="full_ft lora_diffusion weight_lora adapters bitfit"
NUM_SEEDS=10
START_SEED=42
OUTPUT_DIR="./outputs/multi_seed_experiments"
CONFIG="configs/base_config.yaml"

echo "Configuration:"
echo "  Tasks: $TASKS"
echo "  Methods: $METHODS"
echo "  Seeds: $START_SEED to $((START_SEED + NUM_SEEDS - 1)) ($NUM_SEEDS total)"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: Run training experiments
echo "Step 1: Running training experiments..."
echo "=========================================="
python scripts/run_experiments.py \
  --tasks $TASKS \
  --methods $METHODS \
  --num_seeds $NUM_SEEDS \
  --start_seed $START_SEED \
  --output_dir $OUTPUT_DIR \
  --config $CONFIG

if [ $? -ne 0 ]; then
    echo "Error: Training experiments failed"
    exit 1
fi

echo ""
echo "Training complete!"
echo ""

# Step 2: Run evaluation on all checkpoints (token-level accuracy, same metric as training; no generation)
echo "Step 2: Running evaluation (token-level, same as training)..."
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
              --output_file "$checkpoint_dir/eval_results.json" || true
        else
            echo "Warning: Checkpoint not found: $checkpoint_dir"
        fi
    done
done

echo ""
echo "Token-level evaluation complete!"
echo ""

# Step 3: Run classification head evaluation (optional; not used for main tables)
echo "Step 3: Running classification-head evaluation (optional)..."
echo "=========================================="

for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
    for method in $METHODS; do
        checkpoint_dir="$OUTPUT_DIR/sst2_${method}_seed${seed}"
        
        if [ -d "$checkpoint_dir" ]; then
            echo "Classification head: sst2_${method}_seed${seed}"
            python scripts/evaluate.py \
              --checkpoint "$checkpoint_dir" \
              --task sst2 \
              --split validation \
              --eval_classification_head \
              --output_file "$checkpoint_dir/eval_results_ch.json" || true
        fi
    done
done

echo ""
echo "Classification-head evaluation complete!"
echo ""

# Step 4: Collect results and compute statistics
echo "Step 4: Collecting results and computing statistics..."
echo "=========================================="

python scripts/collect_results_with_stats.py \
  --base_dir $OUTPUT_DIR \
  --task sst2 \
  --methods $METHODS \
  --seeds "$START_SEED-$((START_SEED + NUM_SEEDS - 1))" \
  --output collected_results_with_stats.json \
  --baseline full_ft

if [ $? -ne 0 ]; then
    echo "Error: Results collection failed"
    exit 1
fi

echo ""
echo "Results collected!"
echo ""

# Step 5: Generate paper tables
echo "Step 5: Generating paper tables..."
echo "=========================================="

python scripts/generate_paper_tables.py \
  --results collected_results_with_stats.json \
  --output paper_tables_with_stats.tex

if [ $? -ne 0 ]; then
    echo "Error: Table generation failed"
    exit 1
fi

echo ""
echo "Tables generated!"
echo ""

# Summary
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - collected_results_with_stats.json"
echo "  - collected_results_with_stats.summary.txt"
echo "  - paper_tables_with_stats.tex"
echo ""
echo "Next steps:"
echo "  1. Review collected_results_with_stats.summary.txt for statistical summary"
echo "  2. Copy tables from paper_tables_with_stats.tex to doc/Paper.tex"
echo "  3. Compile paper: cd doc && pdflatex Paper.tex"
echo ""
