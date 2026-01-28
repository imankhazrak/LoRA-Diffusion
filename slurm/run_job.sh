#!/bin/bash
#SBATCH --job-name=lora_diffusion
#SBATCH --account=<your_account>          # CHANGE THIS: Your OSC account name
#SBATCH --time=24:00:00                  # Time limit (HH:MM:SS) - adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1                 # Number of GPUs (1-4 typically)
#SBATCH --gpu-type=a100                   # GPU type: a100, v100, or leave empty for default
#SBATCH --cpus-per-task=8                # CPUs per task (adjust based on GPU count)
#SBATCH --mem=128GB                       # Memory (adjust: 64GB for 1 GPU, 128GB for 2, etc.)
#SBATCH --output=logs/slurm-%j.out       # Output log file
#SBATCH --error=logs/slurm-%j.err         # Error log file
#SBATCH --mail-type=BEGIN,END,FAIL        # Email notifications
#SBATCH --mail-user=<your_email>          # CHANGE THIS: Your email address

###############################################################################
# OSC LoRA-Diffusion Job Script
# 
# This script runs LoRA-Diffusion training/evaluation on OSC.
# 
# Usage:
#   sbatch slurm/run_job.sh
# 
# Or with environment variables:
#   TASK=sst2 METHOD=lora_diffusion MAX_STEPS=5000 HOURS=12 sbatch slurm/run_job.sh
#
# Configuration:
#   - Edit the SBATCH directives above to set GPU type, time, memory
#   - Set environment variables before submitting (see below)
#   - Or edit defaults in this script
###############################################################################

# Print job information
echo "=========================================="
echo "LoRA-Diffusion Job Started"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "User: $USER"
echo "Account: $SLURM_JOB_ACCOUNT"
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Load modules (OSC-specific)
echo "Loading modules..."
module purge
module load python/3.10
module load cuda/11.8.0  # Adjust CUDA version as needed
module list
echo ""

# Activate virtual environment
VENV_PATH=${VENV_PATH:-"$HOME/lora-diffusion/venv"}
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please create it with: python -m venv $VENV_PATH"
    exit 1
fi

echo "Activating virtual environment: $VENV_PATH"
source $VENV_PATH/bin/activate

# Verify Python and packages
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo ""

# Set environment variables for OSC scratch space
export SCRATCH=/fs/scratch/users/$USER
export HF_HOME=$SCRATCH/lora-diffusion/data/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/lora-diffusion/data/transformers_cache
export TORCH_HOME=$SCRATCH/lora-diffusion/data/torch_cache

# Create necessary directories
mkdir -p $SCRATCH/lora-diffusion/{data/{cache,hf_cache,transformers_cache,torch_cache},outputs,checkpoints,logs}
mkdir -p logs

# Navigate to project directory
PROJECT_DIR=${PROJECT_DIR:-"$HOME/lora-diffusion"}
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found at $PROJECT_DIR"
    exit 1
fi

cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

echo "Project directory: $PROJECT_DIR"
echo "Scratch directory: $SCRATCH/lora-diffusion"
echo ""

# ============================================================================
# CONFIGURATION - Set defaults or use environment variables
# ============================================================================

# Task and method
TASK=${TASK:-"sst2"}                    # Task: sst2, squad, xsum, agnews
METHOD=${METHOD:-"lora_diffusion"}      # Method: lora_diffusion, weight_lora, full_ft, etc.
SEED=${SEED:-42}                        # Random seed

# Training parameters
MAX_STEPS=${MAX_STEPS:-5000}             # Maximum training steps
BATCH_SIZE=${BATCH_SIZE:-}               # Batch size (empty = use config default)
LEARNING_RATE=${LEARNING_RATE:-}         # Learning rate (empty = use config default)

# Job type
JOB_TYPE=${JOB_TYPE:-"train"}            # train or evaluate
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}     # Required for evaluation

# Output configuration
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRATCH/lora-diffusion/outputs/${TASK}_${METHOD}_${SLURM_JOB_ID}"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$SCRATCH/lora-diffusion/checkpoints/${TASK}_${METHOD}_${SLURM_JOB_ID}"}

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR

echo "=========================================="
echo "Job Configuration"
echo "=========================================="
echo "Task: $TASK"
echo "Method: $METHOD"
echo "Seed: $SEED"
echo "Max Steps: $MAX_STEPS"
echo "Job Type: $JOB_TYPE"
echo "Output Directory: $OUTPUT_DIR"
echo "Checkpoint Directory: $CHECKPOINT_DIR"
echo ""

# ============================================================================
# TASK-SPECIFIC CONFIGURATION
# ============================================================================

# Set task-specific max steps if not provided
if [ -z "$MAX_STEPS" ] || [ "$MAX_STEPS" == "auto" ]; then
    case $TASK in
        sst2|agnews)
            MAX_STEPS=5000
            ;;
        squad)
            MAX_STEPS=10000
            ;;
        xsum)
            MAX_STEPS=15000
            ;;
        *)
            MAX_STEPS=10000
            ;;
    esac
    echo "Auto-set MAX_STEPS to $MAX_STEPS for task $TASK"
fi

# ============================================================================
# BUILD COMMAND
# ============================================================================

CMD="python scripts/${JOB_TYPE}.py"

if [ "$JOB_TYPE" == "train" ]; then
    # Training command
    CMD="$CMD --config configs/base_config.yaml"
    CMD="$CMD --task $TASK"
    CMD="$CMD --method $METHOD"
    CMD="$CMD --seed $SEED"
    CMD="$CMD --max_steps $MAX_STEPS"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    
    # Add overrides
    CMD="$CMD --overrides"
    CMD="$CMD data.cache_dir=$SCRATCH/lora-diffusion/data/cache"
    CMD="$CMD output.checkpoint_dir=$CHECKPOINT_DIR"
    
    # Optional overrides
    if [ ! -z "$BATCH_SIZE" ]; then
        CMD="$CMD training.batch_size=$BATCH_SIZE"
    fi
    if [ ! -z "$LEARNING_RATE" ]; then
        CMD="$CMD training.learning_rate=$LEARNING_RATE"
    fi
    
elif [ "$JOB_TYPE" == "evaluate" ]; then
    # Evaluation command
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "ERROR: CHECKPOINT_PATH must be set for evaluation"
        exit 1
    fi
    
    CMD="$CMD --checkpoint $CHECKPOINT_PATH"
    CMD="$CMD --task $TASK"
    CMD="$CMD --split test"
    CMD="$CMD --output_file $OUTPUT_DIR/test_results.json"
    CMD="$CMD --device cuda"
    
    if [ ! -z "$BATCH_SIZE" ]; then
        CMD="$CMD --batch_size $BATCH_SIZE"
    fi
else
    echo "ERROR: Unknown JOB_TYPE: $JOB_TYPE (must be 'train' or 'evaluate')"
    exit 1
fi

# ============================================================================
# RUN COMMAND
# ============================================================================

echo "=========================================="
echo "Running Command"
echo "=========================================="
echo "$CMD"
echo ""

# Run with error handling
set -e  # Exit on error
START_TIME=$(date +%s)

eval $CMD

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# ============================================================================
# POST-PROCESSING
# ============================================================================

echo ""
echo "=========================================="
echo "Job Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Exit code: $EXIT_CODE"
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Copy results to home directory (optional)
if [ $EXIT_CODE -eq 0 ] && [ "$COPY_RESULTS" == "true" ]; then
    RESULTS_DIR=$HOME/lora-diffusion/results/${TASK}_${METHOD}_${SLURM_JOB_ID}
    mkdir -p $RESULTS_DIR
    echo "Copying results to: $RESULTS_DIR"
    cp -r $OUTPUT_DIR/* $RESULTS_DIR/ 2>/dev/null || true
    echo "Results copied successfully"
fi

# Print final GPU usage
echo ""
echo "Final GPU Usage:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

exit $EXIT_CODE
