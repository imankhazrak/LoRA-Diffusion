#!/bin/bash
#SBATCH --job-name=mt_table
#SBATCH --account=pcs0229
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/multitask_table_only-%j.out
#SBATCH --error=logs/multitask_table_only-%j.err
#
# Run only multi-task table evaluation (assumes outputs/quick_multitask already has
# trained LoRA and router). Use when training was done on cluster and you need to
# fill multitask_composition_results.json and Paper.tex Table 12.
# Usage: sbatch slurm/run_multitask_table_only.slurm

set -e
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  $(date)"

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

BASE_DIR="${BASE_DIR:-outputs/quick_multitask}"
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
mkdir -p logs

if [ -d "$VENV_PATH" ]; then
    echo "Activating venv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi
echo "Python: $(which python)  $(python --version 2>&1)"

echo "Running table evaluation only. BASE_DIR=$BASE_DIR"
python scripts/run_multitask_table.py \
  --sst2_ckpt "$BASE_DIR/sst2_lora_diffusion" \
  --mrpc_ckpt "$BASE_DIR/mrpc_lora_diffusion" \
  --qnli_ckpt "$BASE_DIR/qnli_lora_diffusion" \
  --router_path "$BASE_DIR/composition_router/router.pt" \
  --output "$BASE_DIR/multitask_composition_results.json" \
  --update_paper

echo "Done. $(date)"
