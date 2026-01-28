#!/bin/bash
#SBATCH --job-name=lora_diffusion
#SBATCH --account=<your_account>    # CHANGE THIS
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your_email>    # CHANGE THIS

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load modules (adjust as needed for your cluster)
module purge
module load python/3.10
module load cuda/11.8.0
module list

# Activate virtual environment
source $HOME/lora-diffusion/venv/bin/activate

# Set environment variables
export SCRATCH=/fs/scratch/users/$USER
export HF_HOME=$SCRATCH/lora-diffusion/data/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/lora-diffusion/data/transformers_cache
export TORCH_HOME=$SCRATCH/lora-diffusion/data/torch_cache

# Navigate to project directory
cd $HOME/lora-diffusion

# Set PYTHONPATH
export PYTHONPATH=$HOME/lora-diffusion:$PYTHONPATH

# Default values (can be overridden)
TASK=${TASK:-"sst2"}
METHOD=${METHOD:-"lora_diffusion"}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-5000}

# Create output directory
OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/${TASK}_${METHOD}_${SLURM_JOB_ID}
mkdir -p $OUTPUT_DIR

# Run training
echo "Training $METHOD on $TASK with seed $SEED"
python scripts/train.py \
    --config configs/base_config.yaml \
    --task $TASK \
    --method $METHOD \
    --seed $SEED \
    --max_steps $MAX_STEPS \
    --output_dir $OUTPUT_DIR \
    --overrides \
        "data.cache_dir=$SCRATCH/lora-diffusion/data/cache" \
        "output.checkpoint_dir=$SCRATCH/lora-diffusion/checkpoints/${TASK}_${METHOD}_${SLURM_JOB_ID}"

# Print completion info
echo "End time: $(date)"
echo "Output directory: $OUTPUT_DIR"

# Copy results to home directory (optional)
RESULTS_DIR=$HOME/lora-diffusion/results/${TASK}_${METHOD}_${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR
cp -r $OUTPUT_DIR/* $RESULTS_DIR/
echo "Results copied to: $RESULTS_DIR"
