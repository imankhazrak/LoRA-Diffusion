# Next Steps: Getting Your Job Running

## Current Status ✅

- ✅ Code is ready and tested (42 tests passing)
- ✅ Virtual environment created
- ✅ Job script configured
- ✅ Cache directory issues fixed
- ⚠️ Network timeout when downloading from HuggingFace

## Recommended Action Plan

### Step 1: Pre-download Models (5-10 minutes)

Run this on a **login node** (not in a SLURM job) to download models:

```bash
cd ~/LoRA-Diffusion
./prepare_models.sh
```

Or manually:
```bash
cd ~/LoRA-Diffusion
source venv/bin/activate

# Set cache directories
export HF_HOME=$HOME/LoRA-Diffusion/data/hf_cache
export TRANSFORMERS_CACHE=$HOME/LoRA-Diffusion/data/transformers_cache
export HF_DATASETS_CACHE=$HOME/LoRA-Diffusion/data/datasets_cache

# Download tokenizer
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"

# Download datasets (small samples to test)
python -c "from datasets import load_dataset; load_dataset('glue', 'sst2', split='train[:100]')"
```

### Step 2: Submit Test Job

Once models are cached, submit a quick test:

```bash
cd ~/LoRA-Diffusion
TASK=sst2 \
METHOD=lora_diffusion \
MAX_STEPS=100 \
sbatch slurm/run_job.sh
```

### Step 3: Monitor Job

```bash
# Check status
squeue -u $USER

# Watch logs
tail -f logs/slurm-<job_id>.out

# Check for errors
cat logs/slurm-<job_id>.err
```

### Step 4: Verify Results

After job completes:

```bash
./verify_results.sh \
  outputs/sst2_lora_diffusion_<job_id> \
  checkpoints/sst2_lora_diffusion_<job_id>
```

## Alternative: Submit Without Pre-download

If you want to try submitting again (network might be better now):

```bash
cd ~/LoRA-Diffusion
TASK=sst2 METHOD=lora_diffusion MAX_STEPS=100 sbatch slurm/run_job.sh
```

The job will download models during execution. If network is stable, it should work.

## What's Fixed

1. ✅ Virtual environment path detection
2. ✅ Cache directory configuration
3. ✅ Python script cache setup
4. ✅ Account and email configured
5. ✅ All results saving verified

## If Job Still Fails

Check the error log:
```bash
cat logs/slurm-<job_id>.err
```

Common issues:
- **Network timeout**: Pre-download models (Step 1)
- **Module not found**: Check `module avail python`
- **Permission denied**: Check directory permissions

## Full Training Run

Once test works, submit full training:

```bash
# SST-2 (12 hours)
TASK=sst2 METHOD=lora_diffusion MAX_STEPS=5000 sbatch slurm/run_job.sh

# SQuAD (24 hours)
TASK=squad METHOD=lora_diffusion MAX_STEPS=10000 sbatch slurm/run_job.sh
```

## Summary

**Immediate next step**: Run `./prepare_models.sh` to pre-download models, then submit a test job.

Everything else is ready! 🚀
