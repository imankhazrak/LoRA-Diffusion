#!/bin/bash
# Re-run only collect_results_with_stats and generate_paper_tables.
# Use this after the main multi-seed job (and optionally rerun_adapters_seed42) have finished,
# so all 50 runs exist and you get full 10-seed stats.

set -e

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/multi_seed_experiments}"
METHODS="full_ft lora_diffusion weight_lora adapters bitfit"
SEEDS="42-51"

echo "Collecting results from: $OUTPUT_DIR"
echo ""

python scripts/collect_results_with_stats.py \
  --base_dir "$OUTPUT_DIR" \
  --task sst2 \
  --methods $METHODS \
  --seeds "$SEEDS" \
  --output collected_results_with_stats.json \
  --baseline full_ft

python scripts/generate_paper_tables.py \
  --results collected_results_with_stats.json \
  --output paper_tables_with_stats.tex

echo ""
echo "Done. See collected_results_with_stats.json and paper_tables_with_stats.tex"
