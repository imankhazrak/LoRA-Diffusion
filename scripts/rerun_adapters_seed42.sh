#!/bin/bash
# Re-run only the failed sst2_adapters_seed42, then re-evaluate and re-collect stats
# so you have a full 10 seeds for adapters. Run from project root.

set -e

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"
CONFIG="configs/base_config.yaml"
METHODS="full_ft lora_diffusion weight_lora adapters bitfit"
SEEDS="42-51"

echo "=========================================="
echo "Re-run: sst2_adapters_seed42 + re-collect stats"
echo "=========================================="
echo "Output base: $OUTPUT_DIR"
echo ""

# 1. Train only sst2_adapters_seed42
echo "Step 1: Training sst2_adapters_seed42..."
python scripts/train.py \
  --config $CONFIG \
  --task sst2 \
  --method adapters \
  --seed 42 \
  --output_dir "$OUTPUT_DIR/sst2_adapters_seed42"

echo "  Done."
echo ""

# 2. Eval (generation-based)
echo "Step 2: Evaluation (generation)..."
python scripts/evaluate.py \
  --checkpoint "$OUTPUT_DIR/sst2_adapters_seed42" \
  --task sst2 \
  --split validation \
  --output_file "$OUTPUT_DIR/sst2_adapters_seed42/eval_results.json"
echo "  Done."
echo ""

# 3. Eval (classification head)
echo "Step 3: Evaluation (classification head)..."
python scripts/evaluate.py \
  --checkpoint "$OUTPUT_DIR/sst2_adapters_seed42" \
  --task sst2 \
  --split validation \
  --eval_classification_head \
  --output_file "$OUTPUT_DIR/sst2_adapters_seed42/eval_results_ch.json"
echo "  Done."
echo ""

# 4. Re-collect results (all 50 runs now present)
echo "Step 4: Collecting results and computing statistics..."
python scripts/collect_results_with_stats.py \
  --base_dir "$OUTPUT_DIR" \
  --task sst2 \
  --methods $METHODS \
  --seeds "$SEEDS" \
  --output collected_results_with_stats.json \
  --baseline full_ft
echo "  Done."
echo ""

# 5. Regenerate paper tables
echo "Step 5: Generating paper tables..."
python scripts/generate_paper_tables.py \
  --results collected_results_with_stats.json \
  --output paper_tables_with_stats.tex
echo "  Done."
echo ""

echo "=========================================="
echo "Re-run complete. Full 10 seeds for adapters."
echo "=========================================="
