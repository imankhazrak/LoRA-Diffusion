#!/bin/bash
# Re-evaluate all completed multi-seed experiments
# This script finds all experiment directories that have training_summary.json
# and runs evaluation (generation + classification head) on each.

# Don't use set -e: ((fail_count++)) returns exit 1 when fail_count was 0, which would kill the job
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"
TASK="${TASK:-sst2}"

echo "=========================================="
echo "Re-evaluating all completed experiments"
echo "=========================================="
echo "Base directory: $OUTPUT_DIR"
echo "Task: $TASK"
echo ""

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

# Find all experiment directories
experiments=()
for exp_dir in "$OUTPUT_DIR"/sst2_*_seed*; do
    if [ -d "$exp_dir" ] && [ -f "$exp_dir/training_summary.json" ]; then
        experiments+=("$exp_dir")
    fi
done

total=${#experiments[@]}
echo "Found $total completed experiments"
echo ""

if [ $total -eq 0 ]; then
    echo "No completed experiments found. Exiting."
    exit 0
fi

# Counters
success_count=0
fail_count=0
skipped_count=0

# Process each experiment
for exp_dir in "${experiments[@]}"; do
    exp_name=$(basename "$exp_dir")
    echo "----------------------------------------"
    echo "Evaluating: $exp_name"
    echo "----------------------------------------"
    
    # Force re-evaluation (comment out the check below if you want to skip already-evaluated)
    # if [ -f "$exp_dir/eval_results.json" ] && [ -f "$exp_dir/eval_results_ch.json" ]; then
    #     echo "  Already evaluated, skipping..."
    #     ((skipped_count++))
    #     continue
    # fi
    
    # Generation-based evaluation
    echo "  Running generation-based evaluation..."
    if python scripts/evaluate.py \
        --checkpoint "$exp_dir" \
        --task "$TASK" \
        --split validation \
        --output_file "$exp_dir/eval_results.json" \
        --device cuda 2>&1 | tee "$exp_dir/eval_gen.log"; then
        echo "  ✓ Generation eval completed"
    else
        echo "  ✗ Generation eval failed"
        fail_count=$((fail_count + 1))
        continue
    fi
    
    # Classification-head evaluation
    echo "  Running classification-head evaluation..."
    if python scripts/evaluate.py \
        --checkpoint "$exp_dir" \
        --task "$TASK" \
        --split validation \
        --eval_classification_head \
        --output_file "$exp_dir/eval_results_ch.json" \
        --device cuda 2>&1 | tee "$exp_dir/eval_ch.log"; then
        echo "  ✓ Classification-head eval completed"
        success_count=$((success_count + 1))
    else
        echo "  ✗ Classification-head eval failed"
        fail_count=$((fail_count + 1))
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo "Re-evaluation complete"
echo "=========================================="
echo "Total experiments: $total"
echo "Successful: $success_count"
echo "Failed: $fail_count"
echo "Skipped (already done): $skipped_count"
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/collect_results_with_stats.py --base_dir $OUTPUT_DIR --task $TASK --seeds 42-51 --output collected_results_with_stats.json"
echo "  2. Run: python scripts/generate_paper_tables.py --results collected_results_with_stats.json --output paper_tables_with_stats.tex"
