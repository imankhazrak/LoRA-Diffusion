#!/bin/bash
#SBATCH --job-name=quick_mt
#SBATCH --account=pcs0229
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/quick_multitask-%j.out
#SBATCH --error=logs/quick_multitask-%j.err
#
# Quick multi-task Table 12 pipeline: one seed, reduced steps (sanity check).
# Usage: sbatch slurm/run_quick_multitask_table.slurm
# Or from project dir: SEED=42 MAX_STEPS=200 sbatch slurm/run_quick_multitask_table.slurm

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  $(date)"

# Project and venv (match run_job.sh)
if [ -d "$HOME/LoRA-Diffusion" ]; then
    PROJECT_DIR="${PROJECT_DIR:-$HOME/LoRA-Diffusion}"
elif [ -d "$HOME/lora-diffusion" ]; then
    PROJECT_DIR="${PROJECT_DIR:-$HOME/lora-diffusion}"
else
    PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
fi
if [ -d "$HOME/LoRA-Diffusion/venv" ]; then
    VENV_PATH="$HOME/LoRA-Diffusion/venv"
elif [ -d "$HOME/lora-diffusion/venv" ]; then
    VENV_PATH="$HOME/lora-diffusion/venv"
else
    VENV_PATH="${VENV_PATH:-$HOME/LoRA-Diffusion/venv}"
fi

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Optional: use scratch for cache/outputs
if [ -d "/fs/scratch/users/$USER" ] && [ -w "/fs/scratch/users/$USER" ]; then
    export SCRATCH=/fs/scratch/users/$USER
    CACHE_BASE=$SCRATCH/lora-diffusion/data
else
    CACHE_BASE=$PROJECT_DIR/data
fi
export HF_HOME=${HF_HOME:-$CACHE_BASE/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$CACHE_BASE/datasets_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$CACHE_BASE/transformers_cache}
export TORCH_HOME=${TORCH_HOME:-$CACHE_BASE/torch_cache}
mkdir -p "$CACHE_BASE" logs

if [ -d "$VENV_PATH" ]; then
    echo "Activating venv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi
echo "Python: $(which python)  $(python --version 2>&1)"

# Quick-run settings (override with env vars when submitting)
export SEED="${SEED:-42}"
export MAX_STEPS="${MAX_STEPS:-150}"
export ROUTER_STEPS="${ROUTER_STEPS:-80}"
export BASE_DIR="${BASE_DIR:-outputs/quick_multitask}"

echo "SEED=$SEED  MAX_STEPS=$MAX_STEPS  ROUTER_STEPS=$ROUTER_STEPS  BASE_DIR=$BASE_DIR"
bash scripts/run_quick_multitask_table.sh
echo "Done. $(date)"
