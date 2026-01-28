#!/bin/bash
# Quick reference templates for different job configurations
# Copy the relevant template to run_job.sh or use as environment variables

###############################################################################
# TEMPLATE 1: Quick Test (1-2 hours, small task)
###############################################################################
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
# TASK=sst2 MAX_STEPS=100

###############################################################################
# TEMPLATE 2: Small Task - SST-2 or AGNews (4-8 hours)
###############################################################################
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu-type=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
# TASK=sst2 MAX_STEPS=5000

###############################################################################
# TEMPLATE 3: Medium Task - SQuAD (8-16 hours)
###############################################################################
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu-type=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
# TASK=squad MAX_STEPS=10000

###############################################################################
# TEMPLATE 4: Large Task - XSum or Summarization (16-32 hours)
###############################################################################
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu-type=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
# TASK=xsum MAX_STEPS=15000

###############################################################################
# TEMPLATE 5: Multi-GPU Training (2 GPUs)
###############################################################################
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=2
#SBATCH --gpu-type=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
# TASK=squad MAX_STEPS=10000

###############################################################################
# TEMPLATE 6: V100 GPU (if A100 not available)
###############################################################################
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu-type=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
# TASK=sst2 MAX_STEPS=5000

###############################################################################
# TEMPLATE 7: Evaluation Job (quick, 1-2 hours)
###############################################################################
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
# JOB_TYPE=evaluate CHECKPOINT_PATH=/path/to/checkpoint

###############################################################################
# TEMPLATE 8: Long-running Experiment (3 days)
###############################################################################
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu-type=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
# TASK=xsum MAX_STEPS=20000

###############################################################################
# USAGE EXAMPLES
###############################################################################

# Example 1: Quick test
# Edit run_job.sh with Template 1 settings, then:
# sbatch slurm/run_job.sh

# Example 2: Full training with custom settings
# TASK=sst2 METHOD=lora_diffusion MAX_STEPS=5000 \
# sbatch slurm/run_job.sh

# Example 3: Evaluation
# JOB_TYPE=evaluate TASK=sst2 \
# CHECKPOINT_PATH=/fs/scratch/users/$USER/lora-diffusion/checkpoints/... \
# sbatch slurm/run_job.sh

# Example 4: Multiple jobs with different seeds
# for SEED in 42 43 44; do
#     TASK=sst2 SEED=$SEED MAX_STEPS=5000 \
#     sbatch slurm/run_job.sh
# done
