#!/bin/bash
# Run multi-seed experiments for statistical analysis
# (set -e disabled so one failed run does not abort the whole job; we collect partial results)

echo "=========================================="
echo "Multi-Seed Experiment Pipeline"
echo "=========================================="
echo ""

# Configuration (override with env: TASKS, OUTPUT_DIR, etc.)
TASKS="${TASKS:-sst2}"
METHODS="${METHODS:-full_ft lora_diffusion weight_lora adapters bitfit}"
NUM_SEEDS="${NUM_SEEDS:-5}"
START_SEED="${START_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"
CONFIG="${CONFIG:-configs/base_config.yaml}"

echo "Configuration:"
echo "  Tasks: $TASKS"
echo "  Methods: $METHODS"
echo "  Seeds: $START_SEED to $((START_SEED + NUM_SEEDS - 1)) ($NUM_SEEDS total)"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: Run training experiments (continue even if some runs fail; collect will use completed ones)
echo "Step 1: Running training experiments..."
echo "=========================================="
TRAIN_EXIT=0
python scripts/run_experiments.py \
  --tasks $TASKS \
  --methods $METHODS \
  --num_seeds $NUM_SEEDS \
  --start_seed $START_SEED \
  --output_dir $OUTPUT_DIR \
  --config $CONFIG || TRAIN_EXIT=$?

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "Warning: Some training runs failed (check logs). Continuing with evaluation and collect for completed runs."
fi

echo ""
echo "Training step finished."
echo ""

# Step 2: Run evaluation (for classification: token-level + generation so we report task accuracy)
echo "Step 2: Running evaluation..."
echo "=========================================="

for task in $TASKS; do
  for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
    for method in $METHODS; do
        checkpoint_dir="$OUTPUT_DIR/${task}_${method}_seed${seed}"
        if [ -d "$checkpoint_dir" ]; then
            echo "Evaluating: ${task}_${method}_seed${seed}"
            # For classification (sst2, qnli, etc.) pass --run_generation so eval_results.json has generation_accuracy (used as val_accuracy)
            case "$task" in
              sst2|qnli|agnews|mrpc|cola|rte|mnli) EXTRA_EVAL_ARGS="--run_generation" ;;
              *) EXTRA_EVAL_ARGS="" ;;
            esac
            python scripts/evaluate.py \
              --checkpoint "$checkpoint_dir" \
              --task "$task" \
              --split validation \
              $EXTRA_EVAL_ARGS \
              --output_file "$checkpoint_dir/eval_results.json" || true
        else
            echo "Warning: Checkpoint not found: $checkpoint_dir"
        fi
    done
  done
done

echo ""
echo "Evaluation complete!"
echo ""

# Step 3: Run classification head evaluation (optional; not used for main tables)
echo "Step 3: Running classification-head evaluation (optional)..."
echo "=========================================="

for task in $TASKS; do
  for seed in $(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))); do
    for method in $METHODS; do
        checkpoint_dir="$OUTPUT_DIR/${task}_${method}_seed${seed}"
        if [ -d "$checkpoint_dir" ]; then
            echo "Classification head: ${task}_${method}_seed${seed}"
            python scripts/evaluate.py \
              --checkpoint "$checkpoint_dir" \
              --task "$task" \
              --split validation \
              --eval_classification_head \
              --output_file "$checkpoint_dir/eval_results_ch.json" || true
        fi
    done
  done
done

echo ""
echo "Classification-head evaluation complete!"
echo ""

# Step 4 & 5: Collect results and generate tables (per task; do not exit on failure so we get partial outputs)
for task in $TASKS; do
  echo "Step 4: Collecting results for task=$task..."
  echo "=========================================="
  if ! python scripts/collect_results_with_stats.py \
    --base_dir $OUTPUT_DIR \
    --task $task \
    --methods $METHODS \
    --seeds "$START_SEED-$((START_SEED + NUM_SEEDS - 1))" \
    --output "collected_results_with_stats_${task}.json" \
    --baseline full_ft; then
    echo "Warning: Results collection failed for $task (e.g. too few completed runs). Check logs."
  fi

  echo "Step 5: Generating paper tables for task=$task..."
  echo "=========================================="
  if ! python scripts/generate_paper_tables.py \
    --results "collected_results_with_stats_${task}.json" \
    --output "paper_tables_with_stats_${task}.tex"; then
    echo "Warning: Table generation failed for $task."
  fi
  echo ""
done

echo "Results collected and tables generated!"
echo ""

# Summary
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated files (per task):"
for task in $TASKS; do
  echo "  - collected_results_with_stats_${task}.json"
  echo "  - collected_results_with_stats_${task}.summary.txt"
  echo "  - paper_tables_with_stats_${task}.tex"
done
echo ""
echo "Next steps:"
echo "  1. Review collected_results_with_stats_<task>.summary.txt for statistical summary"
echo "  2. Copy tables from paper_tables_with_stats_<task>.tex to doc/Paper.tex"
echo "  3. Compile paper: cd doc && pdflatex Paper.tex"
echo ""
