#!/bin/bash
#SBATCH --job-name=lora_adapters_42
#SBATCH --account=pcs0229
#SBATCH --time=02:00:00              # ~20 min train + eval + collect
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
# Re-run only the failed sst2_adapters_seed42, then re-eval and re-collect
# so you have a full 10 seeds for adapters. Run AFTER the main multi-seed job.
#
# Usage:
#   sbatch slurm/rerun_adapters_seed42.slurm
#
# Uses same OUTPUT_DIR as main pipeline (./outputs/multi_seed_experiments).
###############################################################################

set -e

echo "=========================================="
echo "Re-run: sst2_adapters_seed42 (full 10 seeds)"
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

# Same output base as main pipeline (script default: ./outputs/multi_seed_experiments)
bash scripts/rerun_adapters_seed42.sh

echo ""
echo "End time: $(date)"
echo "Done. Full 10 seeds for adapters in collected_results_with_stats.json"
