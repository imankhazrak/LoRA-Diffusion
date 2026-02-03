#!/bin/bash
#SBATCH --job-name=reeval_all_10s
#SBATCH --account=pcs0229
#SBATCH --time=08:00:00
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
# Re-run val accuracy for ALL 5 models (full_ft, lora_diffusion, weight_lora,
# adapters, bitfit) with 10 seeds (42-51), same as training seeds.
#
# Usage: sbatch slurm/reevaluate_all_models_10seeds.slurm
###############################################################################

set -e

echo "=========================================="
echo "Re-evaluate: all 5 models, seeds 42-51"
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

[ -d "$VENV_PATH" ] || { echo "ERROR: venv not found at $VENV_PATH"; exit 1; }
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

mkdir -p $PROJECT_DIR/logs
mkdir -p logs
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export PYTHONUNBUFFERED=1

if [ -z "$OUTPUT_DIR" ]; then
    if [ -d "$PROJECT_DIR/outputs/multi_seed_experiments" ]; then
        export OUTPUT_DIR="$PROJECT_DIR/outputs/multi_seed_experiments"
    else
        export OUTPUT_DIR="$SCRATCH/lora-diffusion/outputs/multi_seed_experiments"
    fi
fi
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo ""

bash scripts/reevaluate_all_models_10seeds.sh

echo ""
echo "End time: $(date)"
echo "Results: collected_results_with_stats.json, paper_tables_with_stats.tex"
echo ""
