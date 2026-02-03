#!/bin/bash
#SBATCH --job-name=rerun_wl_adapt
#SBATCH --account=pcs0229
#SBATCH --time=96:00:00
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
# Re-run weight_lora and adapters only (10 seeds each) so model.pt is in
# experiment dir and val accuracy is correct. Same approach as BitFit rerun.
#
# Usage: sbatch slurm/rerun_weightlora_adapters.slurm
###############################################################################

set -e

echo "=========================================="
echo "Re-run Weight LoRA + Adapters (10 seeds each)"
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

# Require scratch for outputs (200+ GB); avoid writing to project and hitting quota
if [ "$SCRATCH" = "$PROJECT_DIR" ]; then
    if [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ] && [ -w "$TMPDIR" ]; then
        export SCRATCH="$TMPDIR"
        echo "Using TMPDIR as SCRATCH: $SCRATCH"
    else
        echo "ERROR: Scratch not available (/fs/scratch/users/$USER missing or not writable). Outputs need 200+ GB; cannot use project dir. Set SCRATCH when submitting, e.g.: SCRATCH=/path/to/scratch sbatch slurm/rerun_weightlora_adapters.slurm"
        exit 1
    fi
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

# Use scratch to avoid project disk quota (50 runs need ~300+ GB)
export OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/multi_seed_experiments
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"
echo ""

bash scripts/rerun_weightlora_adapters.sh

echo ""
echo "End time: $(date)"
echo "Checkpoints: $OUTPUT_DIR/sst2_weight_lora_seed*, sst2_adapters_seed*"
echo ""
