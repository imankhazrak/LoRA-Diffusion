# SLURM Job Scripts for OSC

This directory contains SLURM job scripts for running LoRA-Diffusion on the Ohio Supercomputer Center (OSC).

## Quick Start

### 1. Basic Training Job

```bash
# Edit slurm/run_job.sh to set your account and email
# Then submit:
sbatch slurm/run_job.sh
```

### 2. Customized Job Submission

```bash
# Set environment variables and submit
TASK=sst2 \
METHOD=lora_diffusion \
MAX_STEPS=5000 \
HOURS=12 \
sbatch slurm/run_job.sh
```

## Scripts Overview

### `run_job.sh` - Main Configurable Job Script

**Features:**
- Configurable GPU type and count
- Adjustable time limits
- Supports both training and evaluation
- Automatic task-specific configuration
- Environment setup for OSC

**SBATCH Directives (Edit these):**
```bash
#SBATCH --account=<your_account>     # Your OSC account
#SBATCH --time=24:00:00              # Time limit (HH:MM:SS)
#SBATCH --gpus-per-node=1            # Number of GPUs (1-4)
#SBATCH --gpu-type=a100              # GPU type: a100, v100, or empty for default
#SBATCH --cpus-per-task=8            # CPUs (8 per GPU recommended)
#SBATCH --mem=128GB                  # Memory (64GB per GPU recommended)
#SBATCH --mail-user=<your_email>     # Your email
```

**Environment Variables:**
```bash
# Task and method
TASK=sst2                    # Task: sst2, squad, xsum, agnews
METHOD=lora_diffusion        # Method: lora_diffusion, weight_lora, full_ft, etc.
SEED=42                      # Random seed

# Training parameters
MAX_STEPS=5000               # Maximum training steps (or "auto" for task-specific)
BATCH_SIZE=32                # Batch size (optional, uses config default if not set)
LEARNING_RATE=1e-4           # Learning rate (optional)

# Job type
JOB_TYPE=train               # train or evaluate
CHECKPOINT_PATH=/path/to/checkpoint  # Required for evaluation

# Paths (optional)
VENV_PATH=$HOME/lora-diffusion/venv
PROJECT_DIR=$HOME/lora-diffusion
COPY_RESULTS=true            # Copy results to home directory after completion
```

### `train.slurm` - Simple Training Script

Basic training script with minimal configuration. Good for quick tests.

### `sweep_array.slurm` - Batch Experiment Script

Runs multiple experiments in parallel using SLURM job arrays.

## GPU and Time Configuration

### GPU Types on OSC

Common GPU types available:
- `a100` - NVIDIA A100 (40GB or 80GB)
- `v100` - NVIDIA V100 (32GB)
- (empty) - Default GPU (varies by partition)

### Recommended Resource Settings

#### Small Task (SST-2, AGNews)
```bash
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
MAX_STEPS=5000
```

#### Medium Task (SQuAD)
```bash
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
MAX_STEPS=10000
```

#### Large Task (XSum, Summarization)
```bash
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu-type=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
MAX_STEPS=15000
```

#### Multi-GPU Training
```bash
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=2
#SBATCH --gpu-type=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
```

### Time Estimates

Approximate training times on A100 (40GB):
- SST-2: 2-4 hours (5000 steps)
- SQuAD: 4-8 hours (10000 steps)
- XSum: 8-16 hours (15000 steps)

## Usage Examples

### Example 1: Quick Test Run

```bash
# Edit run_job.sh to set account/email, then:
TASK=sst2 \
METHOD=lora_diffusion \
MAX_STEPS=100 \
sbatch slurm/run_job.sh
```

### Example 2: Full Training Run

```bash
TASK=squad \
METHOD=lora_diffusion \
MAX_STEPS=10000 \
BATCH_SIZE=32 \
LEARNING_RATE=2e-4 \
sbatch slurm/run_job.sh
```

### Example 3: Evaluation

```bash
JOB_TYPE=evaluate \
TASK=sst2 \
CHECKPOINT_PATH=/fs/scratch/users/$USER/lora-diffusion/checkpoints/sst2_lora_diffusion_12345/checkpoint-5000 \
sbatch slurm/run_job.sh
```

### Example 4: Multiple Seeds

```bash
# Run with different seeds
for SEED in 42 43 44; do
    TASK=sst2 \
    METHOD=lora_diffusion \
    SEED=$SEED \
    MAX_STEPS=5000 \
    sbatch slurm/run_job.sh
done
```

### Example 5: Method Comparison

```bash
# Compare different methods
for METHOD in lora_diffusion weight_lora full_ft; do
    TASK=sst2 \
    METHOD=$METHOD \
    MAX_STEPS=5000 \
    sbatch slurm/run_job.sh
done
```

## Monitoring Jobs

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Detailed view
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"

# Check specific job
squeue -j <job_id>
```

### View Logs

```bash
# View output log
tail -f logs/slurm-<job_id>.out

# View error log
tail -f logs/slurm-<job_id>.err

# View last 100 lines
tail -n 100 logs/slurm-<job_id>.out
```

### Cancel Job

```bash
scancel <job_id>
```

### Check GPU Usage

```bash
# SSH to compute node (if job is running)
ssh <node_name>

# Check GPU
nvidia-smi
```

## Troubleshooting

### Job Fails Immediately

1. Check error log: `cat logs/slurm-<job_id>.err`
2. Verify virtual environment exists: `ls $HOME/lora-diffusion/venv`
3. Verify project directory: `ls $HOME/lora-diffusion`
4. Check module availability: `module avail python`

### Out of Memory (OOM)

1. Reduce batch size: `BATCH_SIZE=16`
2. Increase memory: `#SBATCH --mem=256GB`
3. Enable gradient checkpointing in config

### Time Limit Exceeded

1. Increase time limit: `#SBATCH --time=48:00:00`
2. Reduce max steps: `MAX_STEPS=3000`
3. Use faster GPU: `#SBATCH --gpu-type=a100`

### Module Not Found

1. Check available modules: `module avail`
2. Update module load commands in script
3. Verify Python version: `module load python/3.10 && python --version`

### Data Download Issues

1. Check scratch space: `df -h $SCRATCH`
2. Verify cache directory permissions
3. Pre-download data locally first

## Best Practices

1. **Start Small**: Test with `MAX_STEPS=100` first
2. **Monitor Resources**: Check GPU utilization with `nvidia-smi`
3. **Use Scratch**: Always use `/fs/scratch/users/$USER` for data/outputs
4. **Save Checkpoints**: Enable checkpointing in config
5. **Email Notifications**: Set up email alerts for job completion
6. **Log Everything**: Check logs regularly for issues
7. **Resource Efficiency**: Request only what you need (don't over-request GPUs/memory)

## OSC-Specific Notes

- Scratch space: `/fs/scratch/users/$USER` (temporary, fast)
- Home space: `$HOME` (persistent, slower)
- GPU partitions: Check with `sinfo` for available GPU partitions
- Account limits: Check with `sacctmgr show assoc user=$USER`

### If you don't receive job emails from OSC

OSC sends job (BEGIN/END/FAIL) emails from **slurm@osc.edu** or **@osc.edu**. If you don't see them:

1. **Check spam/junk** – Institutional filters (e.g. BGSU) often quarantine them.
2. **Whitelist** – Ask IT to allow: `slurm@osc.edu`, `no-reply@osc.edu`, `oschelp@osc.edu`.
3. **Delays** – Many jobs can cause OSC to throttle emails to your address; notifications may arrive late.
4. **Verify** – In your script you have `#SBATCH --mail-user=your@email` and `#SBATCH --mail-type=BEGIN,END,FAIL` (or `ALL`).

**Check status without email** – Replace `JOBID` with your job number (e.g. from `squeue -u $USER`):
```bash
sacct -j JOBID --format=JobID,State,End,ExitCode
tail -f logs/slurm-JOBID.out
```

For help: [OSC Help](https://www.osc.edu/support) or oschelp@osc.edu.

## Support

For OSC-specific issues:
- OSC Help: https://www.osc.edu/support
- OSC Documentation: https://www.osc.edu/documentation

For code issues:
- Check project README.md
- Review test outputs: `pytest tests/ -v`
