#!/bin/bash
# Final script to collect results and update paper tables after job completion

set -e

JOB_ID="${1:-43996576}"
TASK="${2:-sst2}"
BASE_DIR="/users/PCS0229/imankhazrak/LoRA-Diffusion"

cd "$BASE_DIR"

echo "=========================================="
echo "Collecting Results and Updating Paper Tables"
echo "Job ID: $JOB_ID"
echo "Task: $TASK"
echo "=========================================="

# Step 1: Collect all results
echo ""
echo "Step 1: Collecting experiment results..."
python scripts/collect_results.py \
    --parameter-counts parameter_counts.json \
    --task "$TASK" \
    --job-id "$JOB_ID" \
    --output-dir lora-diffusion/outputs \
    --output collected_results_final.json

if [ ! -f "collected_results_final.json" ]; then
    echo "ERROR: Failed to collect results"
    exit 1
fi

echo "✓ Results collected"

# Step 2: Update paper tables
echo ""
echo "Step 2: Updating paper tables..."
python scripts/update_paper_tables.py \
    --collected-results collected_results_final.json \
    --paper-tex doc/Paper.tex \
    --handle-discrepancy \
    --output doc/Paper.tex

if [ ! -f "doc/Paper.tex" ]; then
    echo "ERROR: Failed to update paper"
    exit 1
fi

echo "✓ Paper tables updated"

# Step 3: Print summary
echo ""
echo "=========================================="
echo "Summary of Collected Results"
echo "=========================================="
python -c "
import json
with open('collected_results_final.json') as f:
    data = json.load(f)
    
formatted = data.get('formatted_for_tables', {})
param_counts = data.get('raw_results', {}).get('parameter_counts', {})

print('\nParameter Counts:')
for method in ['full_ft', 'lora_diffusion', 'weight_lora', 'adapters', 'bitfit', 'prefix_tuning']:
    if method in param_counts:
        pc = param_counts[method]
        print(f'  {method:20s} {pc[\"trainable_percent\"]:6.2f}% ({pc[\"trainable_params\"]/1e6:.1f}M params)')

print('\nExperiment Metrics:')
for method in ['full_ft', 'lora_diffusion', 'weight_lora', 'adapters', 'bitfit', 'prefix_tuning']:
    if method in formatted:
        m = formatted[method]
        if m.get('val_accuracy'):
            print(f'  {method:20s} Val Acc: {m[\"val_accuracy\"]:.2f}%')
        if m.get('train_loss'):
            print(f'  {method:20s} Train Loss: {m[\"train_loss\"]:.4f}')
"

echo ""
echo "=========================================="
echo "Done! Paper tables updated in doc/Paper.tex"
echo "=========================================="
