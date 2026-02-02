#!/bin/bash
#SBATCH --job-name=reeval_multi_seed
#SBATCH --account=pcs0229
#SBATCH --time=08:00:00              # 8 hours (previous 4h run timed out at 37/50)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# Re-evaluate all completed multi-seed experiments to get real validation metrics
#
# Usage:
#   sbatch slurm/reevaluate_multi_seed.slurm
###############################################################################

set -e

echo "=========================================="
echo "Re-evaluating Multi-Seed Experiments"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module purge
module load python/3.10
module load cuda/11.8.0

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

source $VENV_PATH/bin/activate

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

mkdir -p logs
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Step 1: Re-evaluate all completed experiments
bash scripts/reevaluate_all_multi_seed.sh

echo ""
echo "=========================================="
echo "Step 2: Collect results and generate paper tables"
echo "=========================================="
bash scripts/collect_and_tables_only.sh || { echo "Warning: collect/tables step had non-zero exit (check logs)"; true; }

echo ""
echo "End time: $(date)"
echo "All steps complete: re-evaluation, collection, paper tables."
