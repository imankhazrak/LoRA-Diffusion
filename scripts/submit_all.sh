#!/bin/bash
# Submit all GLUE experiments (Instruction2.md): single-task then multi-task.
# Optional: set MAX_PARALLEL to cap concurrent jobs; set OUTPUT_DIR, CONFIG.

TASKS="${TASKS:-sst2 qnli mrpc}"
METHODS="${METHODS:-full_ft lora_diffusion weight_lora adapters bitfit}"
SEEDS="${SEEDS:-42 43 44 45 46}"
ENCODER_MODES="${ENCODER_MODES:-frozen}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/glue_single}"
MT_OUTPUT_DIR="${MT_OUTPUT_DIR:-./outputs/glue_multitask}"
CONFIG="${CONFIG:-configs/base_config.yaml}"
mkdir -p logs

echo "=== Single-task jobs (skip if MULTITASK_ONLY=1) ==="
if [ "${MULTITASK_ONLY}" != "1" ]; then
  echo "Tasks: $TASKS | Methods: $METHODS | Seeds: $SEEDS | Modes: $ENCODER_MODES"
  for task in $TASKS; do
    for method in $METHODS; do
      for seed in $SEEDS; do
        for mode in $ENCODER_MODES; do
          echo "Submitting TASK=$task METHOD=$method SEED=$seed ENCODER_MODE=$mode OUTPUT_DIR=$OUTPUT_DIR CONFIG=$CONFIG"
          sbatch --export=TASK=$task,METHOD=$method,SEED=$seed,ENCODER_MODE=$mode,OUTPUT_DIR=$OUTPUT_DIR,CONFIG=$CONFIG slurm/slurm_single_task.slurm
          if [ -n "$MAX_PARALLEL" ]; then
            while [ $(squeue -u $USER -h | wc -l) -ge "$MAX_PARALLEL" ]; do sleep 30; done
          fi
        done
      done
    done
  done
else
  echo "MULTITASK_ONLY=1: skipping single-task jobs."
fi

echo "=== Multi-task jobs (skip if SINGLE_TASK_ONLY=1) ==="
if [ "${SINGLE_TASK_ONLY}" != "1" ]; then
  for method in $METHODS; do
    for seed in $SEEDS; do
      for mode in $ENCODER_MODES; do
        echo "Submitting METHOD=$method SEED=$seed ENCODER_MODE=$mode OUTPUT_DIR=$MT_OUTPUT_DIR"
        sbatch --export=METHOD=$method,SEED=$seed,ENCODER_MODE=$mode,OUTPUT_DIR=$MT_OUTPUT_DIR,CONFIG=$CONFIG slurm/slurm_multitask.slurm
        if [ -n "$MAX_PARALLEL" ]; then
          while [ $(squeue -u $USER -h | wc -l) -ge "$MAX_PARALLEL" ]; do sleep 30; done
        fi
      done
    done
  done
else
  echo "SINGLE_TASK_ONLY=1: skipping multi-task jobs."
fi

echo "Done. Check: squeue -u \$USER"
