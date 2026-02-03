#!/bin/bash
#SBATCH --job-name=data_eff
#SBATCH --account=pcs0229
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --output=logs/slurm-data-efficiency-%j.out
#SBATCH --error=logs/slurm-data-efficiency-%j.err

###############################################################################
# Data-efficiency sweep for Figure 3: train LoRA-Diffusion and Weight LoRA
# at 10%, 20%, 40%, 60%, 80%, 100% of SST-2, collect val accuracy, write
# data_efficiency_results.json to project root.
#
# Usage: sbatch slurm/run_data_efficiency_sweep.slurm
###############################################################################

set -e

echo "=========================================="
echo "Data-efficiency sweep (Figure 3)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [ -d "${PROJECT_DIR}/venv" ]; then
    VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/venv}"
elif [ -d "${PROJECT_DIR}/.venv" ]; then
    VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
else
    VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
fi
[ -d "$VENV_PATH" ] || { echo "ERROR: venv not found at $VENV_PATH"; exit 1; }

module purge
module load python/3.10
module load cuda/11.8.0
source "$VENV_PATH/bin/activate"

if [ -d "/fs/scratch/users/$USER" ] && [ -w "/fs/scratch/users/$USER" ]; then
    SCRATCH_BASE="/fs/scratch/users/$USER/lora-diffusion"
else
    [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ] && [ -w "$TMPDIR" ] && SCRATCH_BASE="$TMPDIR/lora-diffusion" || SCRATCH_BASE="${PROJECT_DIR}/outputs/data_efficiency_sweep"
fi
mkdir -p "$SCRATCH_BASE"
CACHE_BASE="${SCRATCH_BASE}/data"
export HF_HOME="${CACHE_BASE}/hf_cache"
export TRANSFORMERS_CACHE="${CACHE_BASE}/transformers_cache"
export HF_DATASETS_CACHE="${CACHE_BASE}/datasets_cache"
export TORCH_HOME="${CACHE_BASE}/torch_cache"
export HUGGINGFACE_HUB_CACHE="${CACHE_BASE}/hub_cache"
mkdir -p "$CACHE_BASE"/{hf_cache,transformers_cache,datasets_cache,torch_cache,hub_cache}
mkdir -p "${PROJECT_DIR}/logs"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

OUTPUT_DIR="${SCRATCH_BASE}/data_efficiency_sweep"
RESULTS_FILE="${PROJECT_DIR}/data_efficiency_results.json"

echo "Output dir (checkpoints): $OUTPUT_DIR"
echo "Results file: $RESULTS_FILE"
echo ""

python scripts/run_data_efficiency_sweep.py \
  --output_dir "$OUTPUT_DIR" \
  --results_file "$RESULTS_FILE" \
  --seeds 42

echo ""
echo "End: $(date)"
echo "Results: $RESULTS_FILE"
echo "Regenerate figure: python scripts/generate_figures.py"
echo ""
