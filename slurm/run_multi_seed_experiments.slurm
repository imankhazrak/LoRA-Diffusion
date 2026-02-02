#!/bin/bash
#SBATCH --job-name=lora_multi_seed
#SBATCH --account=pcs0229
#SBATCH --time=168:00:00              # 7 days for 50 experiments (10 seeds x 5 methods)
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
# LoRA-Diffusion: Full multi-seed statistical experiment pipeline
#
# Runs: training (10 seeds x 5 methods on SST-2) -> eval -> collect stats -> tables
#
# Usage:
#   sbatch slurm/run_multi_seed_experiments.slurm
#
# Optional env overrides before sbatch:
#   NUM_SEEDS=5 TASKS="sst2 agnews" sbatch slurm/run_multi_seed_experiments.slurm
###############################################################################

set -e

echo "=========================================="
echo "LoRA-Diffusion Multi-Seed Pipeline"
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

# Project and venv (match run_experiments_osc.sh)
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

echo "Activating: $VENV_PATH"
source $VENV_PATH/bin/activate
echo "Python: $(which python)"
echo ""

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

# Outputs on scratch to avoid filling home
export OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/multi_seed_experiments
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the full pipeline (training -> eval -> collect stats -> tables)
bash scripts/run_multi_seed_experiments.sh

echo ""
echo "=========================================="
echo "Pipeline finished"
echo "=========================================="
echo "End time: $(date)"
echo "Results (in project dir): collected_results_with_stats.json, paper_tables_with_stats.tex"
echo "Raw outputs: $OUTPUT_DIR"
echo ""
