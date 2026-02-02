# LoRA-Diffusion: Parameter-Efficient Fine-Tuning via Low-Rank Trajectory Decomposition

Complete implementation of trajectory-level low-rank adaptation for discrete diffusion language models.

## Overview

This codebase implements LoRA-Diffusion, a novel parameter-efficient fine-tuning method that applies low-rank decomposition to the denoising trajectory rather than model weights. Key features:

- **Trajectory-level adaptation**: Modifies denoising path $x_t \to x_{t-1}$ with learned low-rank perturbations
- **Step-adaptive ranks**: r=64/32/8 for early/middle/late diffusion steps
- **Instruction conditioning**: Task-specific perturbations via conditioned up-projection
- **Multi-task support**: Classification, QA, summarization, reasoning, translation, generation
- **Efficient**: 28.7% trainable parameters total (1.1% trajectory adapters only), 95.7% relative to full fine-tuning on SST-2

## Repository Structure

```
lora-diffusion/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── configs/                          # YAML configurations
│   ├── base_config.yaml             # Default hyperparameters
│   ├── tasks/                       # Task-specific configs
│   │   ├── sst2.yaml
│   │   ├── squad.yaml
│   │   └── xsum.yaml
│   └── methods/                     # Method-specific configs
│       ├── lora_diffusion.yaml
│       ├── weight_lora.yaml
│       ├── full_ft.yaml
│       ├── prefix_tuning.yaml
│       ├── adapters.yaml
│       └── bitfit.yaml
├── src/                             # Source code
│   ├── __init__.py
│   ├── models/                      # Model implementations
│   │   ├── __init__.py
│   │   ├── base_diffusion.py       # Base masked diffusion LM
│   │   ├── lora_modules.py         # LoRA-Diffusion modules
│   │   └── baselines.py            # Baseline PEFT methods
│   ├── data/                        # Data loading
│   │   ├── __init__.py
│   │   ├── task_loader.py          # Task-specific loaders
│   │   └── collators.py            # Data collators
│   ├── training/                    # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main trainer
│   │   └── losses.py               # Loss functions
│   ├── evaluation/                  # Evaluation
│   │   ├── __init__.py
│   │   └── metrics.py              # Metric computation
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── config.py               # Config management
│       ├── logging_utils.py        # Logging setup
│       └── checkpoint.py           # Checkpointing
├── scripts/                         # Training/eval scripts
│   ├── train.py                    # Main training script (--seed for reproducibility)
│   ├── evaluate.py                 # Evaluation script (greedy sampling, fixed seq length)
│   ├── run_experiments.py          # Batch experiments
│   ├── analyze_effective_rank.py   # Effective rank from singular value spectra
│   ├── measure_latency.py          # Runtime overhead measurement
│   └── verify_table_consistency.py # Check table numbers vs parameter_counts
├── slurm/                          # SLURM job scripts
│   ├── train.slurm                 # Single training job
│   └── sweep_array.slurm           # Job array for sweeps
├── tests/                          # Unit tests
│   ├── test_lora_modules.py
│   ├── test_diffusion.py
│   └── test_training.py
└── notebooks/                      # Analysis notebooks
    └── analyze_results.ipynb
```

## Installation

### Step 1: Environment Setup

```bash
# Create conda environment
conda create -n lora-diffusion python=3.10
conda activate lora-diffusion

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install package and dependencies
cd lora-diffusion
pip install -e .
```

### Step 2: Verify Installation

```bash
python -c "import torch; import transformers; print('Installation successful!')"
```

## Quick Start

### Minimal Example (Fast Sanity Check)

Run a quick training on a tiny subset:

```bash
python scripts/train.py \
    --config configs/base_config.yaml \
    --task sst2 \
    --method lora_diffusion \
    --max_steps 100 \
    --subset_size 1000 \
    --eval_frequency 50 \
    --output_dir ./outputs/sanity_check
```

### Full Training Example

Train LoRA-Diffusion on SST-2:

```bash
python scripts/train.py \
    --config configs/base_config.yaml \
    --task sst2 \
    --method lora_diffusion \
    --output_dir ./outputs/sst2_lora_diffusion \
    --seed 42
```

### Baseline Comparisons

```bash
# Full fine-tuning
python scripts/train.py --task sst2 --method full_ft --output_dir ./outputs/sst2_full_ft

# Weight LoRA
python scripts/train.py --task sst2 --method weight_lora --output_dir ./outputs/sst2_weight_lora

# Prefix tuning
python scripts/train.py --task sst2 --method prefix_tuning --output_dir ./outputs/sst2_prefix

# Adapters
python scripts/train.py --task sst2 --method adapters --output_dir ./outputs/sst2_adapters

# BitFit
python scripts/train.py --task sst2 --method bitfit --output_dir ./outputs/sst2_bitfit
```

### Multi-Task Training

```bash
# Train on multiple tasks sequentially
python scripts/run_experiments.py \
    --tasks sst2 squad xsum \
    --methods lora_diffusion weight_lora full_ft \
    --seeds 42 43 44 \
    --output_dir ./outputs/multi_task_sweep
```

## Running on OSC (Ohio Supercomputer Center)

### Step 1: Setup on OSC

```bash
# SSH to OSC
ssh username@pitzer.osc.edu

# Clone repository
cd $HOME
git clone <your-repo-url> lora-diffusion
cd lora-diffusion

# Load modules
module load python/3.10
module load cuda/11.8.0

# Create environment
python -m venv venv
source venv/bin/activate
pip install -e .

# Set up directories
export SCRATCH=/fs/scratch/users/$USER
mkdir -p $SCRATCH/lora-diffusion/{data,outputs,checkpoints}
```

### Step 2: Configure Paths

Edit `configs/base_config.yaml` to set OSC paths:

```yaml
data:
  cache_dir: /fs/scratch/users/<username>/lora-diffusion/data
  
output:
  base_dir: /fs/scratch/users/<username>/lora-diffusion/outputs
  checkpoint_dir: /fs/scratch/users/<username>/lora-diffusion/checkpoints
```

### Step 3: Submit Single Job

**Option A: Using the configurable script (Recommended)**

```bash
# 1. Edit slurm/run_job.sh to set your account and email
# 2. Submit with default settings:
sbatch slurm/run_job.sh

# 3. Or customize with environment variables:
TASK=sst2 \
METHOD=lora_diffusion \
MAX_STEPS=5000 \
sbatch slurm/run_job.sh
```

**Option B: Using the simple script**

```bash
# Edit slurm/train.slurm to set your username/email
sbatch slurm/train.slurm
```

### Step 4: Submit Job Array (Multiple Tasks/Seeds)

```bash
# Edit slurm/sweep_array.slurm for your experiments
sbatch slurm/sweep_array.slurm
```

### Step 5: Configure GPU and Time Resources

Edit the SBATCH directives in `slurm/run_job.sh`:

```bash
#SBATCH --time=24:00:00          # Time limit (adjust as needed)
#SBATCH --gpus-per-node=1        # Number of GPUs
#SBATCH --gpu-type=a100          # GPU type: a100, v100, or empty
#SBATCH --cpus-per-task=8        # CPUs (8 per GPU recommended)
#SBATCH --mem=128GB              # Memory (64GB per GPU recommended)
```

See `slurm/README_SLURM.md` for detailed configuration options and templates.

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f slurm-<jobid>.out

# Check GPU usage
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"
```

## Configuration System

All hyperparameters are managed via YAML configs. Override via command line:

```bash
python scripts/train.py \
    --config configs/base_config.yaml \
    --task sst2 \
    --method lora_diffusion \
    --model.hidden_dim 1024 \
    --model.num_layers 12 \
    --training.learning_rate 2e-4 \
    --training.batch_size 32 \
    --lora.rank_early 64 \
    --lora.rank_mid 32 \
    --lora.rank_late 8
```

### Key Configuration Parameters

**Model Architecture:**
- `model.hidden_dim`: Hidden dimension (default: 768)
- `model.num_layers`: Number of transformer layers (default: 12)
- `model.num_heads`: Number of attention heads (default: 12)
- `model.vocab_size`: Vocabulary size (default: 30522)

**Diffusion Process:**
- `diffusion.num_steps`: Total diffusion steps T (default: 100)
- `diffusion.schedule`: Noise schedule ("linear" or "cosine", default: "cosine")
- `diffusion.forward_type`: Forward corruption ("mask" or "uniform", default: "mask")

**LoRA-Diffusion:**
- `lora.rank_early`: Rank for early steps (default: 64)
- `lora.rank_mid`: Rank for middle steps (default: 32)
- `lora.rank_late`: Rank for late steps (default: 8)
- `lora.num_modules`: Number of modules k (default: 2)
- `lora.scaling_early`: σ for early steps (default: 1.0)
- `lora.scaling_mid`: σ for middle steps (default: 0.5)
- `lora.scaling_late`: σ for late steps (default: 0.25)
- `lora.rank_reg_weight`: λ_rank (default: 0.01)
- `lora.orth_reg_weight`: λ_orth (default: 0.001)

**Training:**
- `training.learning_rate`: Learning rate (default: 1e-4)
- `training.batch_size`: Batch size (default: 64)
- `training.gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `training.max_steps`: Maximum training steps (default: 10000)
- `training.warmup_steps`: Warmup steps (default: 500)
- `training.mixed_precision`: "fp16", "bf16", or "no" (default: "bf16")

## Evaluation

### Evaluate Trained Model

```bash
python scripts/evaluate.py \
    --checkpoint ./outputs/sst2_lora_diffusion/checkpoint-10000 \
    --task sst2 \
    --split test \
    --output_file ./outputs/sst2_lora_diffusion/test_results.json
```

### Compute Metrics

The evaluation script automatically computes task-appropriate metrics:
- **Classification**: Accuracy, F1
- **QA**: Exact Match (EM), Token F1
- **Summarization**: ROUGE-L, BERTScore
- **Generation**: BLEU, ROUGE-L

It also reports efficiency metrics:
- Trainable parameter count and percentage
- Peak GPU memory usage
- Training time per step
- Adapter storage size (MB)

## Reproducing Paper Results

### Main Results (Table 3)

```bash
# Run all methods on all tasks
python scripts/run_experiments.py \
    --tasks sst2 agnews trec squad triviaqa natural_questions gsm8k strategyqa cnn_dm xsum wmt14_ende wmt14_enfr commongen wikibio \
    --methods full_ft weight_lora adapters prefix_tuning bitfit lora_diffusion \
    --seeds 42 43 44 \
    --output_dir ./outputs/main_results
```

### Ablation Studies (Section 5)

**Rank ablation:**
```bash
python scripts/run_experiments.py \
    --tasks sst2 squad xsum \
    --methods lora_diffusion \
    --lora_rank_configs uniform_8 uniform_16 uniform_32 uniform_64 adaptive \
    --output_dir ./outputs/rank_ablation
```

**Number of modules:**
```bash
python scripts/run_experiments.py \
    --tasks sst2 squad xsum \
    --methods lora_diffusion \
    --lora_num_modules 1 2 4 8 \
    --output_dir ./outputs/module_ablation
```

## Extending to New Tasks

1. **Add task configuration** in `configs/tasks/<task_name>.yaml`:

```yaml
task:
  name: my_task
  type: classification  # or qa, summarization, generation
  dataset: my_hf_dataset
  split_names:
    train: train
    validation: validation
    test: test
  max_seq_length: 512
  num_labels: 2  # for classification
```

2. **Add task loader** in `src/data/task_loader.py`:

```python
def load_my_task(cache_dir: str):
    dataset = load_dataset("my_hf_dataset", cache_dir=cache_dir)
    return dataset
```

3. **Run training:**

```bash
python scripts/train.py --task my_task --method lora_diffusion
```

## Performance Tips

### Memory Optimization

1. **Use mixed precision:**
```bash
--training.mixed_precision bf16
```

2. **Gradient accumulation for large effective batch size:**
```bash
--training.batch_size 16 --training.gradient_accumulation_steps 4  # Effective batch size: 64
```

3. **Gradient checkpointing:**
```bash
--model.gradient_checkpointing true
```

### Speed Optimization

1. **Increase workers:**
```bash
--data.num_workers 4
```

2. **Pin memory:**
```bash
--data.pin_memory true
```

3. **Compile model (PyTorch 2.0+):**
```bash
--model.compile true
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `training.batch_size`
- Increase `training.gradient_accumulation_steps`
- Enable `model.gradient_checkpointing`
- Use mixed precision (`training.mixed_precision bf16`)
- Reduce `model.hidden_dim` or `model.num_layers`

### Slow Training

- Increase `data.num_workers`
- Check if using GPU: `torch.cuda.is_available()`
- Ensure data is cached: check `data.cache_dir`
- Reduce evaluation frequency: `training.eval_frequency`

### NaN Loss

- Lower `training.learning_rate`
- Enable gradient clipping: `training.max_grad_norm 1.0`
- Check for numerical stability in custom losses
- Try different mixed precision setting

### SLURM Job Failures

- Check job logs: `less slurm-<jobid>.out`
- Verify GPU allocation: `nvidia-smi`
- Check disk space: `df -h $SCRATCH`
- Ensure modules loaded: `module list`

## Testing

Run unit tests:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_lora_modules.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Citation

If you use this code, please cite:

```bibtex
@article{khazrak2025lora,
  title={LoRA-Diffusion: Parameter-Efficient Fine-Tuning via Low-Rank Trajectory Decomposition},
  author={Khazrak, Iman and Green, Robert},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

- Iman Khazrak (ikhazra@bgsu.edu)
- Robert Green (greenr@bgsu.edu)


## Acknowledgments

This research was conducted at Bowling Green State University. Computational resources provided by Ohio Supercomputer Center.
