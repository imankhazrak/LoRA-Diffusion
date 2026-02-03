#!/bin/bash
#SBATCH --job-name=reg_ablation
#SBATCH --account=pcs0229
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/slurm-reg-ablation-%j.out
#SBATCH --error=logs/slurm-reg-ablation-%j.err

# =============================================================================
# LoRA-Diffusion Regularizer Ablation Runner
# -----------------------------------------------------------------------------
# This job trains/evaluates LoRA-Diffusion on SST-2 for the four regularizer
# settings needed for Table 18:
#   1) baseline        : λ_rank=0.01, λ_orth=0.001
#   2) no_rank         : λ_rank=0,    λ_orth=0.001
#   3) no_orth         : λ_rank=0.01, λ_orth=0
#   4) both_off        : λ_rank=0,    λ_orth=0
#
# Usage:
#   sbatch slurm/run_reg_ablation.sh
#
# Optional environment overrides before sbatch:
#   SEED=42 MAX_STEPS=10000 sbatch slurm/run_reg_ablation.sh
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "LoRA-Diffusion Regularizer Ablation"
echo "Job ID : ${SLURM_JOB_ID:-N/A}"
echo "Node   : ${SLURM_NODELIST:-unknown}"
echo "Start  : $(date)"
echo "=========================================="
echo

# -----------------------------------------------------------------------------
# Locate project root & virtual environment
# -----------------------------------------------------------------------------
# Under SLURM, use submit dir (where sbatch was run) so we don't resolve to /var/spool/slurmd
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
# Prefer venv in project (LoRA-Diffusion/venv or .venv)
if [ -d "${PROJECT_DIR}/venv" ]; then
    VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/venv}"
elif [ -d "${PROJECT_DIR}/.venv" ]; then
    VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
else
    VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: virtual environment not found at $VENV_PATH"
    echo "Create it first (python -m venv .venv && .venv/bin/pip install -r requirements.txt)."
    exit 1
fi

# -----------------------------------------------------------------------------
# Modules & environment
# -----------------------------------------------------------------------------
module purge
module load python/3.10
module load cuda/11.8.0

source "$VENV_PATH/bin/activate"

echo "Python : $(which python)"
python --version
echo

# Scratch/cache setup (falls back to project dir if $SCRATCH unavailable)
if [ -d "/fs/scratch/users/${USER}" ] && [ -w "/fs/scratch/users/${USER}" ]; then
    SCRATCH_BASE="/fs/scratch/users/${USER}/lora-diffusion"
else
    SCRATCH_BASE="${PROJECT_DIR}/scratch"
fi

CACHE_BASE="${SCRATCH_BASE}/data"
export HF_HOME="${CACHE_BASE}/hf_cache"
export TRANSFORMERS_CACHE="${CACHE_BASE}/transformers_cache"
export HF_DATASETS_CACHE="${CACHE_BASE}/datasets_cache"
export TORCH_HOME="${CACHE_BASE}/torch_cache"
export HUGGINGFACE_HUB_CACHE="${CACHE_BASE}/hub_cache"

mkdir -p "$CACHE_BASE"/{hf_cache,transformers_cache,datasets_cache,torch_cache,hub_cache}
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${SCRATCH_BASE}/regularizer_ablation"

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# Experiment hyper-parameters
# -----------------------------------------------------------------------------
TASK="${TASK:-sst2}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-5000}"
EVAL_FREQ="${EVAL_FREQ:-500}"
SAVE_FREQ="${SAVE_FREQ:-1000}"

declare -a REG_CONFIGS=(
  "name=baseline rank=0.01 orth=0.001"
  "name=no_rank rank=0 orth=0.001"
  "name=no_orth rank=0.01 orth=0"
  "name=both_off rank=0 orth=0"
)

SUMMARY_ROWS=()

# -----------------------------------------------------------------------------
# Helper: aggregate metrics after each run
# -----------------------------------------------------------------------------
aggregate_results () {
  local RUN_DIR="$1"
  local SUMMARY_PATH="$RUN_DIR/summary.json"
  local TRAIN_SUMMARY="$RUN_DIR/output/training_summary.json"
  local EVAL_RESULTS="$RUN_DIR/eval_results.json"

  python - <<PY
import json, pathlib, sys
train_path = pathlib.Path("$TRAIN_SUMMARY")
eval_path = pathlib.Path("$EVAL_RESULTS")
if not train_path.exists() or not eval_path.exists():
    sys.exit(0)
with train_path.open() as f:
    train_data = json.load(f)
with eval_path.open() as f:
    eval_data = json.load(f)
metrics = {
    "config_name": "${CONFIG_NAME}",
    "rank_reg_weight": ${RANK_REG},
    "orth_reg_weight": ${ORTH_REG},
    "train_loss": train_data.get("final_training_metrics", {}).get("loss"),
    "val_accuracy": eval_data.get("generation_accuracy", eval_data.get("accuracy")),
    "teacher_forced_accuracy": eval_data.get("accuracy"),
    "evaluation_file": str(eval_path),
    "training_summary": str(train_path),
}
with open("$SUMMARY_PATH", "w") as out_f:
    json.dump(metrics, out_f, indent=2)
PY

  if [ -f "$SUMMARY_PATH" ]; then
      SUMMARY_ROWS+=("$SUMMARY_PATH")
  fi
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
for CONFIG in "${REG_CONFIGS[@]}"; do
    IFS=' ' read -r -a PARTS <<< "$CONFIG"
    declare -A CFG_MAP=()
    for ITEM in "${PARTS[@]}"; do
        KEY="${ITEM%%=*}"
        VALUE="${ITEM#*=}"
        CFG_MAP[$KEY]="$VALUE"
    done
    CONFIG_NAME="${CFG_MAP[name]}"
    RANK_REG="${CFG_MAP[rank]}"
    ORTH_REG="${CFG_MAP[orth]}"

    RUN_ROOT="${SCRATCH_BASE}/regularizer_ablation/${TASK}_lora_reg_${CONFIG_NAME}_seed${SEED}"
    OUTPUT_DIR="${RUN_ROOT}/output"
    CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
    LOG_DIR="${RUN_ROOT}/logs"
    mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

    echo "----------------------------------------"
    echo "Config : $CONFIG_NAME (λ_rank=$RANK_REG, λ_orth=$ORTH_REG)"
    echo "Output : $RUN_ROOT"
    echo "----------------------------------------"

    # ------------------------ Training ------------------------
    python scripts/train.py \
        --config configs/base_config.yaml \
        --task "$TASK" \
        --method lora_diffusion \
        --seed "$SEED" \
        --max_steps "$MAX_STEPS" \
        --output_dir "$OUTPUT_DIR" \
        --overrides \
            "lora.rank_reg_weight=${RANK_REG}" \
            "lora.orth_reg_weight=${ORTH_REG}" \
            "output.checkpoint_dir=${CHECKPOINT_DIR}" \
            "training.eval_frequency=${EVAL_FREQ}" \
            "training.save_frequency=${SAVE_FREQ}" \
            "data.cache_dir=${CACHE_BASE}" \
        > "${LOG_DIR}/training.log" 2>&1

    # Figure out checkpoint to evaluate
    if [ -d "${CHECKPOINT_DIR}/final_model" ]; then
        CKPT_TO_EVAL="${CHECKPOINT_DIR}/final_model"
    else
        CKPT_TO_EVAL=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "checkpoint-*" -type d | sort -V | tail -1)
    fi

    if [ -z "${CKPT_TO_EVAL:-}" ] || [ ! -d "$CKPT_TO_EVAL" ]; then
        echo "WARNING: No checkpoint found for $CONFIG_NAME; skipping evaluation."
        continue
    fi

    # ------------------------ Evaluation ------------------------
    python scripts/evaluate.py \
        --checkpoint "$CKPT_TO_EVAL" \
        --task "$TASK" \
        --split validation \
        --device cuda \
        --run_generation \
        --eval_classification_head \
        --output_file "${RUN_ROOT}/eval_results.json" \
        > "${LOG_DIR}/evaluation.log" 2>&1

    aggregate_results "$RUN_ROOT"
done

# -----------------------------------------------------------------------------
# Combine summaries for convenience
# -----------------------------------------------------------------------------
if [ "${#SUMMARY_ROWS[@]}" -gt 0 ]; then
    COMBINED="${SCRATCH_BASE}/regularizer_ablation/regularizer_ablation_summary.json"
    python - <<PY
import json, pathlib
paths = ${SUMMARY_ROWS[@]}
summary = []
for p in paths:
    path = pathlib.Path(p)
    if path.exists():
        with path.open() as f:
            summary.append(json.load(f))
with open("$COMBINED", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Combined summary saved to {COMBINED}")
PY
fi

echo
echo "Completed at $(date)"
