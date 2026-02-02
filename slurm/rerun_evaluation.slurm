#!/bin/bash
#SBATCH --job-name=rerun_eval
#SBATCH --account=pcs0229
#SBATCH --time=02:00:00
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
# Rerun Evaluation for All Methods
# 
# This script reruns evaluation for all trained models from job 43996576
# to get proper validation accuracies with fixed evaluation script.
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "Rerunning Evaluation for All Methods"
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
# Rerun Evaluation for All Methods
# ============================================================================
echo "=========================================="
echo "Evaluating Models"
echo "=========================================="

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
    echo "Evaluating $METHOD"
    echo "Checkpoint: $LATEST_CHECKPOINT"
    echo "----------------------------------------"
    
    EVAL_OUTPUT=$OUTPUT_DIR/eval_results.json
    EVAL_LOG=$OUTPUT_DIR/evaluation_rerun.log
    
    python scripts/evaluate.py \
        --checkpoint $LATEST_CHECKPOINT \
        --task $TASK \
        --split validation \
        --output_file $EVAL_OUTPUT \
        --device cuda \
        --batch_size 32 \
        > $EVAL_LOG 2>&1
    
    if [ -f "$EVAL_OUTPUT" ]; then
        echo "✓ Evaluation results: $EVAL_OUTPUT"
        # Show accuracy
        ACCURACY=$(python -c "import json; d=json.load(open('$EVAL_OUTPUT')); print(f\"{d['metrics']['accuracy']:.4f}\")" 2>/dev/null || echo "N/A")
        echo "  Accuracy: $ACCURACY"
    else
        echo "✗ Evaluation failed for $METHOD"
        echo "  Check log: $EVAL_LOG"
        tail -20 $EVAL_LOG || true
    fi
done

# ============================================================================
# Collect and Update Results
# ============================================================================
echo ""
echo "=========================================="
echo "Collecting Results and Updating Paper Tables"
echo "=========================================="

python scripts/collect_results.py \
    --parameter-counts parameter_counts.json \
    --task $TASK \
    --job-id $ORIGINAL_JOB_ID \
    --output-dir lora-diffusion/outputs \
    --output collected_results_final.json

if [ -f "collected_results_final.json" ]; then
    echo "✓ Results collected"
    
    python scripts/update_paper_tables.py \
        --collected-results collected_results_final.json \
        --paper-tex doc/Paper.tex \
        --handle-discrepancy
    
    echo "✓ Paper tables updated"
    
    # Print summary
    echo ""
    echo "Summary of Results:"
    python -c "
import json
with open('collected_results_final.json') as f:
    data = json.load(f)
formatted = data.get('formatted_for_tables', {})
for method in ['full_ft', 'lora_diffusion', 'weight_lora', 'adapters', 'bitfit']:
    if method in formatted:
        m = formatted[method]
        if m.get('val_accuracy'):
            print(f'  {method:20s} Val Acc: {m[\"val_accuracy\"]:.2f}%')
" || true
else
    echo "✗ Failed to collect results"
fi

echo ""
echo "=========================================="
echo "Evaluation Rerun Complete"
echo "=========================================="
echo "End time: $(date)"
