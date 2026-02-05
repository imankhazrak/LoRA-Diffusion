#!/bin/bash
#SBATCH --job-name=mrpc_multi
#SBATCH --account=pcs0229
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --output=logs/slurm-mrpc-%j.out
#SBATCH --error=logs/slurm-mrpc-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# Full MRPC experiment: same pipeline as SST-2 (10 seeds x 5 methods)
# Training -> eval (token + classification head) -> collect stats -> paper tables
#
# Usage: sbatch slurm/run_multi_seed_mrpc.slurm
###############################################################################

set -e

echo "=========================================="
echo "MRPC Multi-Seed Pipeline (Step 1 replicate)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Module load may be unavailable or return non-zero on some clusters; do not fail the job
module purge 2>/dev/null || true
module load python/3.10 2>/dev/null || true
module load cuda/11.8.0 2>/dev/null || true

# Prefer submission directory (where sbatch was run) so job finds project on cluster
if [ -n "${SLURM_SUBMIT_DIR}" ] && [ -d "${SLURM_SUBMIT_DIR}/venv" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
    VENV_PATH="$SLURM_SUBMIT_DIR/venv"
elif [ -d "$HOME/LoRA-Diffusion/venv" ]; then
    VENV_PATH="$HOME/LoRA-Diffusion/venv"
    PROJECT_DIR="$HOME/LoRA-Diffusion"
elif [ -d "$HOME/lora-diffusion/venv" ]; then
    VENV_PATH="$HOME/lora-diffusion/venv"
    PROJECT_DIR="$HOME/lora-diffusion"
else
    PROJECT_DIR="${PROJECT_DIR:-$HOME/LoRA-Diffusion}"
    VENV_PATH="${VENV_PATH:-$PROJECT_DIR/venv}"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: venv not found at $VENV_PATH (PROJECT_DIR=$PROJECT_DIR, SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})"
    exit 1
fi
source $VENV_PATH/bin/activate

# Prefer scratch to avoid filling home disk quota; fall back to node $TMPDIR then project
USE_TMPDIR=0
if [ -d "/fs/scratch/users/$USER" ] && [ -w "/fs/scratch/users/$USER" ]; then
    export SCRATCH=/fs/scratch/users/$USER
    echo "Using scratch: $SCRATCH"
elif [ -n "${SCRATCH}" ] && [ -d "$SCRATCH" ] && [ -w "$SCRATCH" ]; then
    echo "Using SCRATCH env: $SCRATCH"
elif [ -n "${TMPDIR}" ] && [ -d "$TMPDIR" ] && [ -w "$TMPDIR" ]; then
    export SCRATCH="$TMPDIR"
    USE_TMPDIR=1
    echo "Using TMPDIR (node-local); results will be copied to project at end: $SCRATCH"
else
    export SCRATCH=$PROJECT_DIR
    echo "WARNING: No scratch or TMPDIR; using PROJECT_DIR. Ensure sufficient disk quota."
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

# MRPC: one task, same methods and seeds as SST-2
export TASKS=mrpc
export OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/multi_seed_mrpc
mkdir -p $OUTPUT_DIR
echo "Tasks: $TASKS"
echo "Output directory: $OUTPUT_DIR"
echo ""

bash scripts/run_multi_seed_experiments.sh

# If we used TMPDIR for outputs, copy back to project so results persist
if [ "$USE_TMPDIR" = "1" ] && [ -d "$OUTPUT_DIR" ]; then
    DEST="$PROJECT_DIR/outputs/multi_seed_mrpc"
    echo "Copying outputs from $OUTPUT_DIR to $DEST ..."
    mkdir -p "$DEST"
    rsync -a --info=progress2 "$OUTPUT_DIR/" "$DEST/" 2>/dev/null || cp -r "$OUTPUT_DIR"/* "$DEST/" 2>/dev/null || true
    echo "Copy done."
fi

echo ""
echo "=========================================="
echo "MRPC pipeline finished"
echo "=========================================="
echo "End time: $(date)"
echo "Results (in project dir): collected_results_with_stats_mrpc.json, paper_tables_with_stats_mrpc.tex"
echo "Raw outputs: $OUTPUT_DIR"
echo ""
