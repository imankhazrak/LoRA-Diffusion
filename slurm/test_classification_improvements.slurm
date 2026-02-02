#!/bin/bash
#SBATCH --job-name=test_class_improve
#SBATCH --account=pcs0229
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# Test Classification Improvements and Complete Building Process
# 
# This script:
# 1. Tests the new greedy sampling and classification-specific improvements
# 2. Re-runs evaluation for all methods with improved methodology
# 3. Collects results and updates paper tables
# 4. Verifies accuracy improvements from ~50% to 80%+
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "Testing Classification Improvements"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
echo "Loading modules..."
module purge
module load python/3.10
module load cuda/11.8.0
module list
echo ""

# Find and activate virtual environment
if [ -d "$HOME/LoRA-Diffusion/venv" ]; then
    VENV_PATH="$HOME/LoRA-Diffusion/venv"
    PROJECT_DIR="$HOME/LoRA-Diffusion"
elif [ -d "$HOME/lora-diffusion/venv" ]; then
    VENV_PATH="$HOME/lora-diffusion/venv"
    PROJECT_DIR="$HOME/lora-diffusion"
else
    VENV_PATH=${VENV_PATH:-"$HOME/LoRA-Diffusion/venv"}
    PROJECT_DIR=${PROJECT_DIR:-"$HOME/LoRA-Diffusion"}
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

echo "Activating virtual environment: $VENV_PATH"
source $VENV_PATH/bin/activate

# Verify Python
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo ""

# Set up directories
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Create logs directory
mkdir -p logs

# Configuration
ORIGINAL_JOB_ID="43996576"
TASK="sst2"
SEED=42
METHODS=("full_ft" "lora_diffusion" "weight_lora" "adapters" "bitfit")

echo "Original job ID: $ORIGINAL_JOB_ID"
echo "Task: $TASK"
echo "Methods to evaluate: ${METHODS[@]}"
echo ""

# ============================================================================
# Step 1: Verify Code Changes
# ============================================================================
echo "=========================================="
echo "Step 1: Verifying Code Changes"
echo "=========================================="

echo "Checking for greedy sampling support..."
python -c "
import sys
sys.path.insert(0, '.')
from src.models.base_diffusion import MaskedDiffusionTransformer
import inspect

# Check sample method signature
sig = inspect.signature(MaskedDiffusionTransformer.sample)
if 'greedy' in sig.parameters:
    print('✓ sample() method has greedy parameter')
else:
    print('✗ sample() method missing greedy parameter')
    sys.exit(1)

# Check sample_classification method
if hasattr(MaskedDiffusionTransformer, 'sample_classification'):
    print('✓ sample_classification() method exists')
else:
    print('✗ sample_classification() method missing')
    sys.exit(1)
" || {
    echo "ERROR: Code changes not properly implemented"
    exit 1
}

echo "✓ Code changes verified"
echo ""

# ============================================================================
# Step 2: Re-run Evaluation with Improved Methodology
# ============================================================================
echo "=========================================="
echo "Step 2: Re-running Evaluation with Greedy Sampling"
echo "=========================================="

RESULTS_SUMMARY=()
for METHOD in "${METHODS[@]}"; do
    CHECKPOINT_DIR="$PROJECT_DIR/lora-diffusion/checkpoints/${TASK}_${METHOD}_seed${SEED}_${ORIGINAL_JOB_ID}"
    OUTPUT_DIR="$PROJECT_DIR/lora-diffusion/outputs/${TASK}_${METHOD}_seed${SEED}_${ORIGINAL_JOB_ID}"
    
    # Find latest checkpoint
    LATEST_CHECKPOINT=$(find $CHECKPOINT_DIR -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "⚠ No checkpoint found for $METHOD, skipping"
        continue
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "Evaluating $METHOD with improved methodology"
    echo "Checkpoint: $LATEST_CHECKPOINT"
    echo "----------------------------------------"
    
    EVAL_OUTPUT=$OUTPUT_DIR/eval_results_improved.json
    EVAL_LOG=$OUTPUT_DIR/evaluation_improved.log
    
    # Run evaluation with improved methodology (greedy sampling, fixed seq_len)
    python scripts/evaluate.py \
        --checkpoint $LATEST_CHECKPOINT \
        --task $TASK \
        --split validation \
        --output_file $EVAL_OUTPUT \
        --device cuda \
        --batch_size 32 \
        > $EVAL_LOG 2>&1
    
    if [ -f "$EVAL_OUTPUT" ]; then
        echo "✓ Evaluation completed: $EVAL_OUTPUT"
        # Extract accuracy
        ACCURACY=$(python -c "
import json
try:
    with open('$EVAL_OUTPUT', 'r') as f:
        data = json.load(f)
    acc = data.get('metrics', {}).get('accuracy', 0.0)
    print(f'{acc:.4f}')
except Exception as e:
    print('0.0000')
" 2>/dev/null || echo "0.0000")
        
        ACCURACY_PCT=$(python -c "print(f'{float(\"$ACCURACY\") * 100:.2f}')" 2>/dev/null || echo "0.00")
        echo "  Accuracy: $ACCURACY_PCT%"
        RESULTS_SUMMARY+=("$METHOD:$ACCURACY_PCT")
        
        # Check if accuracy improved significantly
        if (( $(echo "$ACCURACY > 0.70" | bc -l) )); then
            echo "  ✓ Significant improvement detected (>70%)"
        elif (( $(echo "$ACCURACY > 0.60" | bc -l) )); then
            echo "  ⚠ Moderate improvement (60-70%)"
        else
            echo "  ⚠ Low accuracy (<60%), may need further investigation"
        fi
    else
        echo "✗ Evaluation failed for $METHOD"
        echo "  Check log: $EVAL_LOG"
        tail -30 $EVAL_LOG || true
    fi
done

# ============================================================================
# Step 3: Collect Results
# ============================================================================
echo ""
echo "=========================================="
echo "Step 3: Collecting Results"
echo "=========================================="

python scripts/collect_results.py \
    --parameter-counts parameter_counts.json \
    --task $TASK \
    --job-id $ORIGINAL_JOB_ID \
    --output-dir lora-diffusion/outputs \
    --output collected_results_improved.json

if [ -f "collected_results_improved.json" ]; then
    echo "✓ Results collected: collected_results_improved.json"
    
    # Print summary
    echo ""
    echo "Results Summary:"
    python -c "
import json
with open('collected_results_improved.json', 'r') as f:
    data = json.load(f)
formatted = data.get('formatted_for_tables', {})
for method in ['full_ft', 'lora_diffusion', 'weight_lora', 'adapters', 'bitfit']:
    if method in formatted:
        m = formatted[method]
        val_acc = m.get('val_accuracy', 0.0)
        if isinstance(val_acc, (int, float)) and val_acc > 0:
            print(f'  {method:20s} Val Acc: {val_acc:.2f}%')
" || true
else
    echo "✗ Failed to collect results"
    exit 1
fi

# ============================================================================
# Step 4: Update Paper Tables
# ============================================================================
echo ""
echo "=========================================="
echo "Step 4: Updating Paper Tables"
echo "=========================================="

python scripts/update_paper_tables.py \
    --collected-results collected_results_improved.json \
    --paper-tex doc/Paper.tex \
    --handle-discrepancy

if [ $? -eq 0 ]; then
    echo "✓ Paper tables updated"
else
    echo "⚠ Warning: Paper table update had issues (may be non-critical)"
fi

# ============================================================================
# Step 5: Generate Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "Step 5: Final Summary"
echo "=========================================="

echo "Classification Improvement Test Results:"
echo "----------------------------------------"
for RESULT in "${RESULTS_SUMMARY[@]}"; do
    METHOD=$(echo $RESULT | cut -d: -f1)
    ACC=$(echo $RESULT | cut -d: -f2)
    echo "  $METHOD: $ACC%"
done

echo ""
echo "Expected improvements:"
echo "  - Full FT: ~51% → 80%+"
echo "  - LoRA-Diffusion: ~49% → 78%+"
echo ""

# Check if improvements were achieved
IMPROVEMENT_DETECTED=false
for RESULT in "${RESULTS_SUMMARY[@]}"; do
    ACC=$(echo $RESULT | cut -d: -f2)
    if (( $(echo "$ACC > 70.0" | bc -l) )); then
        IMPROVEMENT_DETECTED=true
        break
    fi
done

if [ "$IMPROVEMENT_DETECTED" = true ]; then
    echo "✓ Significant accuracy improvements detected!"
    echo "  Classification improvements are working correctly."
else
    echo "⚠ Accuracy improvements may need further investigation"
    echo "  Check evaluation logs for details"
fi

echo ""
echo "=========================================="
echo "Building Process Complete"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "  1. Review collected_results_improved.json"
echo "  2. Check updated doc/Paper.tex tables"
echo "  3. Compile paper PDF to verify table updates"
echo ""
