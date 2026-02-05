# How to Run Multi-Task Composition (Table 12) Experiments

Everything is in place to run the full pipeline. Follow these steps from the **project root**.

## Prerequisites

- Python env with dependencies (`requirements.txt`).
- GPU recommended for training and evaluation.
- First run will download **SetFit/mrpc** and **SetFit/qnli** (and SST-2 if needed) via HuggingFace.

## Step 1: Train single-task LoRA-Diffusion (SST-2, MRPC, QNLI)

Train one checkpoint per task (same method as existing SST-2 runs):

```bash
# From project root
python scripts/train.py --task sst2 --method lora_diffusion
python scripts/train.py --task mrpc --method lora_diffusion
python scripts/train.py --task qnli --method lora_diffusion
```

Or with explicit output dirs and seed:

```bash
python scripts/train.py --task sst2 --method lora_diffusion --output_dir outputs/sst2_lora_diffusion --seed 42
python scripts/train.py --task mrpc --method lora_diffusion --output_dir outputs/mrpc_lora_diffusion --seed 42
python scripts/train.py --task qnli --method lora_diffusion --output_dir outputs/qnli_lora_diffusion --seed 42
```

Each run should produce a directory containing `model.pt`, `lora_module.pt`, and `config.json`.

## Step 2: Train the composition router

Using the three checkpoint directories from Step 1:

```bash
python scripts/train_composition_router.py \
  --task_checkpoints outputs/sst2_lora_diffusion outputs/mrpc_lora_diffusion outputs/qnli_lora_diffusion \
  --task_names sst2 mrpc qnli \
  --output_dir outputs/composition_router \
  --max_steps 2000 \
  --seed 42
```

This writes `outputs/composition_router/router.pt`.

## Step 3: Run evaluations and fill Table 12

```bash
python scripts/run_multitask_table.py \
  --sst2_ckpt outputs/sst2_lora_diffusion \
  --mrpc_ckpt outputs/mrpc_lora_diffusion \
  --qnli_ckpt outputs/qnli_lora_diffusion \
  --router_path outputs/composition_router/router.pt \
  --output multitask_composition_results.json \
  --update_paper
```

- Results are saved to `multitask_composition_results.json`.
- With `--update_paper`, **Table 12** in `doc/Paper.tex` is updated with the new numbers.

## Optional

- **Without training the router:** Omit `--router_path`. The "Composed (router)" column will still run but will use an untrained router (or `task_modules[0]/router.pt` if present).
- **Config:** Override with `--config path/to/config.yaml` in both router training and table script if needed.
- **SLURM:** You can wrap Step 1–2 in job scripts that call the same commands; run Step 3 after both training jobs complete.

## Quick sanity check

From project root:

```bash
# Quick data load test (no training)
python -c "
from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.data import get_task_loader
from src.utils import load_config
from transformers import AutoTokenizer
cfg = load_config(Path('configs/base_config.yaml'), task_name='mrpc', method_name='lora_diffusion')
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
ds = get_task_loader('mrpc', 'train', cfg['data']['cache_dir'], cfg['task'], tok, max_samples=10)
print('MRPC train samples:', len(ds))
"
```

If this prints `MRPC train samples: 10`, MRPC loading is working.
