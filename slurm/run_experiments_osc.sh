#!/bin/bash
#SBATCH --job-name=lora_diffusion_exp
#SBATCH --account=pcs0229                 # OSC account
#SBATCH --time=48:00:00                   # 48 hours for full experiments
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# LoRA-Diffusion Experiment Runner for OSC
# 
# This script runs experiments to address reviewer concerns:
# 1. Parameter counting
# 2. SST-2 experiments with fixed evaluation (all baselines)
# 3. Multi-task experiments (if time/compute allows)
#
# Usage:
#   sbatch slurm/run_experiments_osc.sh
#
# Or customize:
#   TASK=sst2 METHOD=lora_diffusion MAX_STEPS=5000 sbatch slurm/run_experiments_osc.sh
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "LoRA-Diffusion Experiments Started"
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
    echo "Please create it with: python -m venv $VENV_PATH"
    exit 1
fi

echo "Activating virtual environment: $VENV_PATH"
source $VENV_PATH/bin/activate

# Verify Python
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo ""

# Set up directories
if [ -d "/fs/scratch/users/$USER" ] && [ -w "/fs/scratch/users/$USER" ]; then
    export SCRATCH=/fs/scratch/users/$USER
else
    export SCRATCH=$PROJECT_DIR
fi

CACHE_BASE=$SCRATCH/lora-diffusion/data
export HF_HOME=$CACHE_BASE/hf_cache
export TRANSFORMERS_CACHE=$CACHE_BASE/transformers_cache
export HF_DATASETS_CACHE=$CACHE_BASE/datasets_cache
export TORCH_HOME=$CACHE_BASE/torch_cache
export HUGGINGFACE_HUB_CACHE=$CACHE_BASE/hub_cache

mkdir -p $CACHE_BASE/{cache,hf_cache,transformers_cache,datasets_cache,torch_cache,hub_cache}
mkdir -p $PROJECT_DIR/{outputs,checkpoints,logs}
mkdir -p logs

cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

echo "Project directory: $PROJECT_DIR"
echo "Scratch directory: $SCRATCH/lora-diffusion"
echo ""

# ============================================================================
# STEP 1: Parameter Counting
# ============================================================================
echo "=========================================="
echo "STEP 1: Counting Parameters"
echo "=========================================="

PARAM_COUNTS_FILE=$PROJECT_DIR/parameter_counts.json
python scripts/count_parameters.py \
    --config configs/base_config.yaml \
    --task sst2 \
    --output $PARAM_COUNTS_FILE \
    --methods full_ft lora_diffusion weight_lora adapters bitfit prefix_tuning

echo "Parameter counts saved to: $PARAM_COUNTS_FILE"
cat $PARAM_COUNTS_FILE
echo ""

# ============================================================================
# STEP 2: Run SST-2 Experiments
# ============================================================================
echo "=========================================="
echo "STEP 2: Running SST-2 Experiments"
echo "=========================================="

TASK=sst2
SEED=42
MAX_STEPS=${MAX_STEPS:-5000}  # Use config default or override

# Methods to run
METHODS=("full_ft" "lora_diffusion" "weight_lora" "adapters" "bitfit")

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Training $METHOD on $TASK"
    echo "----------------------------------------"
    
    OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/${TASK}_${METHOD}_seed${SEED}_${SLURM_JOB_ID}
    CHECKPOINT_DIR=$SCRATCH/lora-diffusion/checkpoints/${TASK}_${METHOD}_seed${SEED}_${SLURM_JOB_ID}
    
    mkdir -p $OUTPUT_DIR
    mkdir -p $CHECKPOINT_DIR
    
    python scripts/train.py \
        --config configs/base_config.yaml \
        --task $TASK \
        --method $METHOD \
        --seed $SEED \
        --max_steps $MAX_STEPS \
        --output_dir $OUTPUT_DIR \
        --overrides \
            "data.cache_dir=$CACHE_BASE/cache" \
            "output.checkpoint_dir=$CHECKPOINT_DIR" \
        > $OUTPUT_DIR/training.log 2>&1
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $METHOD completed successfully"
        echo "  Output: $OUTPUT_DIR"
        echo "  Checkpoint: $CHECKPOINT_DIR"
    else
        echo "✗ $METHOD failed with exit code $EXIT_CODE"
        echo "  Check log: $OUTPUT_DIR/training.log"
    fi
done

# ============================================================================
# STEP 3: Evaluate All Models
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 3: Evaluating Models"
echo "=========================================="

for METHOD in "${METHODS[@]}"; do
    CHECKPOINT_DIR=$SCRATCH/lora-diffusion/checkpoints/${TASK}_${METHOD}_seed${SEED}_${SLURM_JOB_ID}
    
    # Find latest checkpoint
    LATEST_CHECKPOINT=$(find $CHECKPOINT_DIR -name "checkpoint-*" -type d | sort -V | tail -1)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "⚠ No checkpoint found for $METHOD, skipping evaluation"
        continue
    fi
    
    echo ""
    echo "Evaluating $METHOD from $LATEST_CHECKPOINT"
    
    OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/${TASK}_${METHOD}_seed${SEED}_${SLURM_JOB_ID}
    EVAL_OUTPUT=$OUTPUT_DIR/eval_results.json
    
    python scripts/evaluate.py \
        --checkpoint $LATEST_CHECKPOINT \
        --task $TASK \
        --split validation \
        --output_file $EVAL_OUTPUT \
        --device cuda \
        > $OUTPUT_DIR/evaluation.log 2>&1
    
    if [ -f "$EVAL_OUTPUT" ]; then
        echo "✓ Evaluation results: $EVAL_OUTPUT"
        cat $EVAL_OUTPUT
    else
        echo "✗ Evaluation failed for $METHOD"
    fi
done

# ============================================================================
# STEP 4: Collect Results Summary
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 4: Collecting Results Summary"
echo "=========================================="

SUMMARY_FILE=$PROJECT_DIR/experiment_summary_${SLURM_JOB_ID}.json
python - << EOF
import json
import os
from pathlib import Path

results = {
    "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
    "task": "$TASK",
    "seed": $SEED,
    "max_steps": $MAX_STEPS,
    "methods": {}
}

methods = ["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit"]
scratch = os.environ.get("SCRATCH", "$PROJECT_DIR")

for method in methods:
    output_dir = f"{scratch}/lora-diffusion/outputs/{TASK}_{{method}}_seed{SEED}_{{os.environ.get('SLURM_JOB_ID', 'unknown')}}"
    eval_file = Path(output_dir) / "eval_results.json"
    
    method_results = {
        "output_dir": output_dir,
        "eval_file": str(eval_file),
        "has_eval_results": eval_file.exists()
    }
    
    if eval_file.exists():
        with open(eval_file) as f:
            eval_data = json.load(f)
            method_results["metrics"] = eval_data.get("metrics", {})
    
    results["methods"][method] = method_results

with open("$SUMMARY_FILE", "w") as f:
    json.dump(results, f, indent=2)

print(f"Summary saved to: $SUMMARY_FILE")
EOF

echo ""
echo "=========================================="
echo "Experiments Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Summary file: $SUMMARY_FILE"
echo ""
echo "Results locations:"
for METHOD in "${METHODS[@]}"; do
    OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/${TASK}_${METHOD}_seed${SEED}_${SLURM_JOB_ID}
    echo "  $METHOD: $OUTPUT_DIR"
done
echo ""
echo "Final GPU Usage:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
