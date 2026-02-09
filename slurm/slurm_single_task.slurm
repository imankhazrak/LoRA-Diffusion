#!/bin/bash
#SBATCH --job-name=glue_st
#SBATCH --account=pcs0229
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=logs/single_%x_%j.out
#SBATCH --error=logs/single_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

# One (task, method, seed, encoder_mode) per job.
# TASK=sst2 METHOD=lora_diffusion SEED=42 ENCODER_MODE=frozen sbatch slurm/slurm_single_task.slurm

set -e
TASK=${TASK:-sst2}
METHOD=${METHOD:-lora_diffusion}
SEED=${SEED:-42}
ENCODER_MODE=${ENCODER_MODE:-frozen}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/glue_single}
CONFIG=${CONFIG:-configs/base_config.yaml}

echo "Job ID: $SLURM_JOB_ID Task=$TASK Method=$METHOD Seed=$SEED"
module purge
module load python/3.10
module load cuda/11.8.0
[ -d "$HOME/LoRA-Diffusion/venv" ] && source "$HOME/LoRA-Diffusion/venv/bin/activate" || source "${VENV_PATH:-$HOME/LoRA-Diffusion/venv}/bin/activate"
PROJECT_DIR=${PROJECT_DIR:-$HOME/LoRA-Diffusion}
if [ -d "/fs/scratch/users/$USER" ] && [ -w "/fs/scratch/users/$USER" ]; then
  export SCRATCH=/fs/scratch/users/$USER
else
  export SCRATCH=$PROJECT_DIR
fi
CACHE_BASE=$SCRATCH/lora-diffusion/data
export HF_HOME=$CACHE_BASE/hf_cache
export TRANSFORMERS_CACHE=$CACHE_BASE/transformers_cache
export TORCH_HOME=$CACHE_BASE/torch_cache
mkdir -p $CACHE_BASE/hf_cache $CACHE_BASE/transformers_cache $CACHE_BASE/torch_cache
# Use scratch for outputs when available to avoid filling home quota
if [ "$SCRATCH" != "$PROJECT_DIR" ]; then
  OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/glue_single
fi
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
RUN_DIR=$OUTPUT_DIR/${TASK}_${METHOD}_seed${SEED}_${ENCODER_MODE}
mkdir -p $RUN_DIR
ENC_OVERRIDE="instruction_encoder_frozen=true"
[ "$ENCODER_MODE" = "train" ] && ENC_OVERRIDE="instruction_encoder_frozen=false"
python scripts/train.py --config $CONFIG --task $TASK --method $METHOD --seed $SEED --output_dir $RUN_DIR --overrides $ENC_OVERRIDE
echo "Output: $RUN_DIR"
