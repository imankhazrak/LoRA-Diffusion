#!/bin/bash
#SBATCH --job-name=recollect_tables
#SBATCH --account=pcs0229
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# Recollect results and generate paper tables only (no re-evaluation).
# Use when eval_results.json already exist and only the val metric definition
# or collection logic changed (e.g. val_accuracy = token-level same as train).
#
# Usage:
#   sbatch slurm/recollect_tables.slurm
###############################################################################

set -e

echo "=========================================="
echo "Recollect Results and Paper Tables"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module purge
module load python/3.10

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

source $VENV_PATH/bin/activate

mkdir -p logs
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

bash scripts/collect_and_tables_only.sh

echo ""
echo "End time: $(date)"
echo "Recollect complete: collected_results_with_stats.json and paper tables."
