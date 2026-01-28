#!/bin/bash
#SBATCH --job-name=lora_sweep
#SBATCH --account=<your_account>    # CHANGE THIS
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --array=0-14              # 15 tasks total (adjust based on experiments)
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your_email>  # CHANGE THIS

# Create logs directory
mkdir -p logs

# Print job information
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load modules
module purge
module load python/3.10
module load cuda/11.8.0

# Activate environment
source $HOME/lora-diffusion/venv/bin/activate

# Set environment variables
export SCRATCH=/fs/scratch/users/$USER
export HF_HOME=$SCRATCH/lora-diffusion/data/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/lora-diffusion/data/transformers_cache
export TORCH_HOME=$SCRATCH/lora-diffusion/data/torch_cache

cd $HOME/lora-diffusion
export PYTHONPATH=$HOME/lora-diffusion:$PYTHONPATH

# Define experiment configurations
# Format: "task method seed"
CONFIGS=(
    # SST-2 experiments
    "sst2 lora_diffusion 42"
    "sst2 weight_lora 42"
    "sst2 full_ft 42"
    "sst2 adapters 42"
    "sst2 bitfit 42"
    
    # SQuAD experiments
    "squad lora_diffusion 42"
    "squad weight_lora 42"
    "squad full_ft 42"
    "squad adapters 42"
    "squad bitfit 42"
    
    # XSum experiments
    "xsum lora_diffusion 42"
    "xsum weight_lora 42"
    "xsum full_ft 42"
    "xsum adapters 42"
    "xsum bitfit 42"
)

# Get configuration for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read -r TASK METHOD SEED <<< "$CONFIG"

echo "Configuration: Task=$TASK, Method=$METHOD, Seed=$SEED"

# Create output directory
OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/${TASK}_${METHOD}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $OUTPUT_DIR

# Set task-specific max steps
if [ "$TASK" == "sst2" ]; then
    MAX_STEPS=5000
elif [ "$TASK" == "squad" ]; then
    MAX_STEPS=10000
elif [ "$TASK" == "xsum" ]; then
    MAX_STEPS=15000
else
    MAX_STEPS=10000
fi

# Run training
echo "Starting training..."
python scripts/train.py \
    --config configs/base_config.yaml \
    --task $TASK \
    --method $METHOD \
    --seed $SEED \
    --max_steps $MAX_STEPS \
    --output_dir $OUTPUT_DIR \
    --overrides \
        "data.cache_dir=$SCRATCH/lora-diffusion/data/cache" \
        "output.checkpoint_dir=$SCRATCH/lora-diffusion/checkpoints/${TASK}_${METHOD}_seed${SEED}"

EXIT_CODE=$?

# Print completion info
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output directory: $OUTPUT_DIR"

# Copy results to home directory
if [ $EXIT_CODE -eq 0 ]; then
    RESULTS_DIR=$HOME/lora-diffusion/results/${TASK}_${METHOD}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
    mkdir -p $RESULTS_DIR
    cp -r $OUTPUT_DIR/* $RESULTS_DIR/
    echo "Results copied to: $RESULTS_DIR"
fi

exit $EXIT_CODE
