#!/usr/bin/env bash
# Quick multi-task Table 12 pipeline: one seed, reduced steps, to verify everything runs.
# Run from project root: bash scripts/run_quick_multitask_table.sh
# On cluster: sbatch slurm/run_quick_multitask_table.slurm
# Requires: Python env with requirements.txt (numpy/scipy compatible; use project venv or cluster modules).

set -e
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-150}"
ROUTER_STEPS="${ROUTER_STEPS:-80}"
BASE_DIR="${BASE_DIR:-outputs/quick_multitask}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
echo "Project root: $ROOT"
echo "Seed: $SEED  Task max_steps: $MAX_STEPS  Router steps: $ROUTER_STEPS  Base dir: $BASE_DIR"

mkdir -p "$BASE_DIR"
export PYTHONPATH="$ROOT:$PYTHONPATH"

# 1) Train single-task LoRA-Diffusion (SST-2, MRPC, QNLI)
for TASK in sst2 mrpc qnli; do
  OUT="$BASE_DIR/${TASK}_lora_diffusion"
  echo "=========================================="
  echo "Training $TASK -> $OUT"
  echo "=========================================="
  python scripts/train.py \
    --task "$TASK" \
    --method lora_diffusion \
    --output_dir "$OUT" \
    --seed "$SEED" \
    --max_steps "$MAX_STEPS" \
    --eval_frequency 50
done

# 2) Train composition router (task run dirs: lora_module.pt in root or final_model/)
SST2_CKPT="$BASE_DIR/sst2_lora_diffusion"
MRPC_CKPT="$BASE_DIR/mrpc_lora_diffusion"
QNLI_CKPT="$BASE_DIR/qnli_lora_diffusion"
ROUTER_DIR="$BASE_DIR/composition_router"
echo "=========================================="
echo "Training router -> $ROUTER_DIR"
echo "=========================================="
python scripts/train_composition_router.py \
  --task_checkpoints "$SST2_CKPT" "$MRPC_CKPT" "$QNLI_CKPT" \
  --task_names sst2 mrpc qnli \
  --output_dir "$ROUTER_DIR" \
  --max_steps "$ROUTER_STEPS" \
  --batch_size 32 \
  --seed "$SEED"

# 3) Run table evaluations and update Paper.tex
echo "=========================================="
echo "Running multi-task table evaluation"
echo "=========================================="
python scripts/run_multitask_table.py \
  --sst2_ckpt "$SST2_CKPT" \
  --mrpc_ckpt "$MRPC_CKPT" \
  --qnli_ckpt "$QNLI_CKPT" \
  --router_path "$ROUTER_DIR/router.pt" \
  --output "$BASE_DIR/multitask_composition_results.json" \
  --update_paper

echo "=========================================="
echo "Quick multi-task run finished successfully."
echo "Results: $BASE_DIR/multitask_composition_results.json"
echo "=========================================="
