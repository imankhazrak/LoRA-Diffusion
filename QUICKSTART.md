# LoRA-Diffusion Quick Start Guide

This guide will get you running LoRA-Diffusion in 5 minutes.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

## Installation (Local)

```bash
# Clone repository
git clone <repo-url> lora-diffusion
cd lora-diffusion

# Create environment
conda create -n lora-diffusion python=3.10
conda activate lora-diffusion

# Install dependencies
pip install -e .
```

## Quick Test (1 minute)

Run a minimal training loop on a tiny subset:

```bash
python scripts/train.py \
    --config configs/base_config.yaml \
    --task sst2 \
    --method lora_diffusion \
    --max_steps 100 \
    --subset_size 500 \
    --output_dir ./outputs/quick_test
```

This should complete in ~1-2 minutes on a GPU and verify your installation works.

## Full Training Example (30 minutes)

Train LoRA-Diffusion on SST-2:

```bash
python scripts/train.py \
    --config configs/base_config.yaml \
    --task sst2 \
    --method lora_diffusion \
    --max_steps 5000 \
    --output_dir ./outputs/sst2_full \
    --seed 42
```

## Compare Methods

```bash
# Train all methods on SST-2
for method in lora_diffusion weight_lora full_ft adapters bitfit; do
    python scripts/train.py \
        --task sst2 \
        --method $method \
        --max_steps 5000 \
        --output_dir ./outputs/sst2_${method}
done
```

## Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint ./outputs/sst2_full/checkpoint-5000 \
    --task sst2 \
    --split validation \
    --output_file ./outputs/sst2_full/eval_results.json
```

## OSC Quick Start

```bash
# SSH to OSC
ssh username@pitzer.osc.edu

# Setup (one-time)
cd $HOME
git clone <repo-url> lora-diffusion
cd lora-diffusion

module load python/3.10
python -m venv venv
source venv/bin/activate
pip install -e .

# Edit SLURM script with your account/email
nano slurm/train.slurm
# Change <your_account> and <your_email>

# Submit job
export TASK=sst2
export METHOD=lora_diffusion
sbatch slurm/train.slurm
```

## Common Issues

### Out of Memory

Reduce batch size:
```bash
python scripts/train.py \
    --task sst2 \
    --method lora_diffusion \
    --overrides training.batch_size=16
```

### Slow Training

Enable mixed precision:
```bash
python scripts/train.py \
    --task sst2 \
    --method lora_diffusion \
    --overrides training.mixed_precision=bf16
```

### Data Not Found

Set cache directory:
```bash
export HF_HOME=/path/to/cache
python scripts/train.py --task sst2 --method lora_diffusion
```

## Next Steps

1. **Try other tasks**: Replace `--task sst2` with `squad` or `xsum`
2. **Tune hyperparameters**: Use `--overrides` to modify config
3. **Run ablations**: See `scripts/run_experiments.py`
4. **Add new tasks**: See "Extending to New Tasks" in README

## Key Files

- `scripts/train.py` - Main training script
- `scripts/evaluate.py` - Evaluation script  
- `configs/base_config.yaml` - Default hyperparameters
- `configs/tasks/*.yaml` - Task-specific configs
- `configs/methods/*.yaml` - Method-specific configs

## Getting Help

- Check logs in `./logs/`
- Run tests: `pytest tests/ -v`
- See full README for detailed documentation
- Report issues on GitHub

## Performance Expectations

On a single NVIDIA A100 (40GB):

| Task | Method | Time | Memory | Performance |
|------|--------|------|--------|-------------|
| SST-2 | LoRA-Diffusion | ~30min | ~8GB | 94%+ accuracy |
| SST-2 | Full FT | ~60min | ~15GB | 95% accuracy |
| SQuAD | LoRA-Diffusion | ~2hrs | ~12GB | 85%+ F1 |
| XSum | LoRA-Diffusion | ~4hrs | ~16GB | 35+ ROUGE-L |

Note: These are approximate values for the 1.3B parameter model configuration.
