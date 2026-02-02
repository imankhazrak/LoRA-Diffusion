#!/bin/bash
#SBATCH --job-name=retrain_peft
#SBATCH --account=pcs0229
#SBATCH --time=96:00:00              # ~4 days for 20 retrains + eval + collect
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
# Retrain Weight LoRA + Adapters, re-eval BitFit, collect + tables
#
# Usage: sbatch slurm/retrain_peft_and_reeval.slurm
###############################################################################

set -e

echo "=========================================="
echo "Retrain PEFT + Re-eval BitFit + Collect"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Load modules
module purge
module load python/3.10
module load cuda/11.8.0

# Project and venv
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

[ -d "$VENV_PATH" ] || { echo "ERROR: venv not found at $VENV_PATH"; exit 1; }
source $VENV_PATH/bin/activate

# Scratch and caches
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
mkdir -p $PROJECT_DIR/logs
mkdir -p logs

cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export PYTHONUNBUFFERED=1

export OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/multi_seed_experiments
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"
echo ""

bash scripts/retrain_peft_and_reeval.sh

echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
echo "End time: $(date)"
echo "Results: collected_results_with_stats.json, paper_tables_with_stats.tex (in project dir)"
echo ""
