# How to Run Jobs on OSC

This guide shows you exactly how to submit and run LoRA-Diffusion jobs on the Ohio Supercomputer Center.

## Prerequisites

1. **OSC Account**: You need an active OSC account
2. **SSH Access**: Be able to SSH to OSC (e.g., `ssh username@pitzer.osc.edu`)
3. **Code Setup**: Have the LoRA-Diffusion code in your home directory
4. **Environment**: Virtual environment created and packages installed

## Quick Start (3 Steps)

### Step 1: Edit the Job Script

```bash
# SSH to OSC
ssh username@pitzer.osc.edu

# Navigate to project
cd ~/lora-diffusion

# Edit the job script
nano slurm/run_job.sh
```

**Change these two lines:**
```bash
#SBATCH --account=<your_account>    # Change <your_account> to your OSC account
#SBATCH --mail-user=<your_email>    # Change <your_email> to your email
```

**Example:**
```bash
#SBATCH --account=PAS1234           # Your OSC account number
#SBATCH --mail-user=john.doe@email.com
```

### Step 2: Configure GPU and Time (Optional)

Edit these lines in `slurm/run_job.sh` based on your needs:

```bash
#SBATCH --time=24:00:00              # Time limit (HH:MM:SS)
#SBATCH --gpus-per-node=1            # Number of GPUs
#SBATCH --gpu-type=a100              # GPU type: a100, v100, or empty
#SBATCH --cpus-per-task=8           # CPUs (8 per GPU recommended)
#SBATCH --mem=128GB                  # Memory (64GB per GPU recommended)
```

**Common configurations:**

| Task | Time | GPU | Memory | Max Steps |
|------|------|-----|--------|-----------|
| Quick test | 2h | 1x A100 | 64GB | 100 |
| SST-2 | 12h | 1x A100 | 64GB | 5000 |
| SQuAD | 24h | 1x A100 | 128GB | 10000 |
| XSum | 48h | 1x A100 | 128GB | 15000 |

### Step 3: Submit the Job

```bash
# Basic submission (uses defaults)
sbatch slurm/run_job.sh

# Or with custom parameters
TASK=sst2 \
METHOD=lora_diffusion \
MAX_STEPS=5000 \
sbatch slurm/run_job.sh
```

## Detailed Examples

### Example 1: Quick Test Run

Test that everything works with a small run:

```bash
# Edit run_job.sh first (account, email, time=02:00:00)
TASK=sst2 \
METHOD=lora_diffusion \
MAX_STEPS=100 \
sbatch slurm/run_job.sh
```

**Expected output:**
```
Submitted batch job 12345678
```

### Example 2: Full Training on SST-2

```bash
# Set time to 12:00:00 in run_job.sh first
TASK=sst2 \
METHOD=lora_diffusion \
MAX_STEPS=5000 \
SEED=42 \
sbatch slurm/run_job.sh
```

### Example 3: Training on SQuAD

```bash
# Set time to 24:00:00 in run_job.sh first
TASK=squad \
METHOD=lora_diffusion \
MAX_STEPS=10000 \
BATCH_SIZE=32 \
sbatch slurm/run_job.sh
```

### Example 4: Training on XSum

```bash
# Set time to 48:00:00 in run_job.sh first
TASK=xsum \
METHOD=lora_diffusion \
MAX_STEPS=15000 \
sbatch slurm/run_job.sh
```

### Example 5: Compare Different Methods

```bash
# Train LoRA-Diffusion
TASK=sst2 METHOD=lora_diffusion MAX_STEPS=5000 sbatch slurm/run_job.sh

# Train Weight LoRA
TASK=sst2 METHOD=weight_lora MAX_STEPS=5000 sbatch slurm/run_job.sh

# Train Full Fine-tuning
TASK=sst2 METHOD=full_ft MAX_STEPS=5000 sbatch slurm/run_job.sh
```

### Example 6: Multiple Seeds

```bash
# Run with different random seeds
for SEED in 42 43 44; do
    TASK=sst2 \
    METHOD=lora_diffusion \
    SEED=$SEED \
    MAX_STEPS=5000 \
    sbatch slurm/run_job.sh
done
```

### Example 7: Evaluation

```bash
# First, find your checkpoint path
# Checkpoints are in: /fs/scratch/users/$USER/lora-diffusion/checkpoints/

JOB_TYPE=evaluate \
TASK=sst2 \
CHECKPOINT_PATH=/fs/scratch/users/$USER/lora-diffusion/checkpoints/sst2_lora_diffusion_12345/checkpoint-5000 \
sbatch slurm/run_job.sh
```

## Monitoring Your Jobs

### Check Job Status

```bash
# List all your jobs
squeue -u $USER

# Detailed view with GPU info
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"

# Check specific job
squeue -j <job_id>
```

**Output example:**
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
12345 gpu          lora_dif user  R       1:23:45     1 node1234
```

### View Job Logs

```bash
# View output log (real-time)
tail -f logs/slurm-<job_id>.out

# View last 100 lines
tail -n 100 logs/slurm-<job_id>.out

# View error log
cat logs/slurm-<job_id>.err

# View full log
less logs/slurm-<job_id>.out
```

### Check Job Details

```bash
# Get detailed job information
scontrol show job <job_id>

# Check job accounting
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

### Cancel a Job

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Finding Your Results

### Output Locations

After job completion, results are in:

```bash
# Main output directory
/fs/scratch/users/$USER/lora-diffusion/outputs/<task>_<method>_<job_id>/

# Checkpoints
/fs/scratch/users/$USER/lora-diffusion/checkpoints/<task>_<method>_<job_id>/

# Logs
~/lora-diffusion/logs/slurm-<job_id>.out
```

### List Recent Outputs

```bash
# List output directories
ls -lht /fs/scratch/users/$USER/lora-diffusion/outputs/ | head -10

# List checkpoints
ls -lht /fs/scratch/users/$USER/lora-diffusion/checkpoints/ | head -10
```

### View Results

```bash
# View training history
cat /fs/scratch/users/$USER/lora-diffusion/outputs/sst2_lora_diffusion_12345/training_history.json

# View evaluation results
cat /fs/scratch/users/$USER/lora-diffusion/outputs/sst2_lora_diffusion_12345/test_results.json
```

## Common Issues and Solutions

### Issue 1: Job Fails Immediately

**Check:**
```bash
# View error log
cat logs/slurm-<job_id>.err

# Common causes:
# - Wrong account name
# - Virtual environment not found
# - Module not available
```

**Solution:**
```bash
# Verify environment exists
ls $HOME/lora-diffusion/venv

# Check modules
module avail python
module avail cuda

# Recreate environment if needed
cd ~/lora-diffusion
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Issue 2: Out of Memory (OOM)

**Solution:**
```bash
# Reduce batch size
BATCH_SIZE=16 sbatch slurm/run_job.sh

# Or increase memory in run_job.sh
#SBATCH --mem=256GB
```

### Issue 3: Time Limit Exceeded

**Solution:**
```bash
# Increase time limit in run_job.sh
#SBATCH --time=48:00:00

# Or reduce max steps
MAX_STEPS=3000 sbatch slurm/run_job.sh
```

### Issue 4: Job Stuck in Queue

**Check:**
```bash
# See why job is waiting
squeue -j <job_id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"

# Common reasons:
# - No GPUs available (wait)
# - Account limits exceeded
# - Wrong partition
```

**Solution:**
```bash
# Check account limits
sacctmgr show assoc user=$USER

# Try different GPU type
# Change in run_job.sh: --gpu-type=v100
```

### Issue 5: Module Not Found

**Solution:**
```bash
# Check available modules
module avail python
module avail cuda

# Update module load commands in run_job.sh
# For example, if python/3.11 is available:
module load python/3.11
```

## Best Practices

1. **Start Small**: Always test with `MAX_STEPS=100` first
2. **Monitor Early**: Check logs in first 10 minutes to catch errors
3. **Use Scratch**: All data/outputs go to `/fs/scratch/users/$USER`
4. **Save Checkpoints**: Enable checkpointing in config
5. **Email Alerts**: Set up email notifications to know when jobs complete
6. **Resource Efficiency**: Don't over-request (request only what you need)
7. **Clean Up**: Remove old outputs/checkpoints to save space

## Quick Reference Commands

```bash
# Submit job
sbatch slurm/run_job.sh

# Check status
squeue -u $USER

# View logs
tail -f logs/slurm-<job_id>.out

# Cancel job
scancel <job_id>

# Check results
ls /fs/scratch/users/$USER/lora-diffusion/outputs/
```

## Getting Help

- **OSC Support**: https://www.osc.edu/support
- **OSC Documentation**: https://www.osc.edu/documentation
- **Project README**: See `README.md` for code-specific help
- **SLURM Docs**: See `slurm/README_SLURM.md` for detailed job configuration

## Example Workflow

Here's a complete workflow from start to finish:

```bash
# 1. SSH to OSC
ssh username@pitzer.osc.edu

# 2. Navigate to project
cd ~/lora-diffusion

# 3. Edit job script (account, email, time)
nano slurm/run_job.sh

# 4. Submit quick test
TASK=sst2 MAX_STEPS=100 sbatch slurm/run_job.sh
# Note job ID: 12345678

# 5. Monitor job
squeue -j 12345678
tail -f logs/slurm-12345678.out

# 6. Once test works, submit full training
TASK=sst2 MAX_STEPS=5000 sbatch slurm/run_job.sh

# 7. Check results when done
ls /fs/scratch/users/$USER/lora-diffusion/outputs/
cat /fs/scratch/users/$USER/lora-diffusion/outputs/sst2_lora_diffusion_*/training_history.json
```

That's it! You're ready to run jobs on OSC.
