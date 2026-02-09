#!/usr/bin/env python3
"""Joint multi-task training on SST2+QNLI+MRPC (one model, shared adapters/LoRA)."""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MaskedDiffusionTransformer, LoRADiffusionModule
from src.models.lora_modules import InstructionEncoder
from src.models.baselines import (
    apply_weight_lora_to_model,
    setup_bitfit,
    apply_adapters_to_model,
)
from src.data import get_task_loader, DiffusionCollator
from src.training import DiffusionTrainer
from src.utils import load_config, setup_logging, set_seed

logger = logging.getLogger(__name__)
TASK_NAMES = ["sst2", "qnli", "mrpc"]


def parse_args():
    p = argparse.ArgumentParser(description="Joint multi-task training (SST2+QNLI+MRPC)")
    p.add_argument("--config", type=str, default="configs/base_config.yaml")
    p.add_argument("--method", type=str, default="lora_diffusion",
                   choices=["lora_diffusion", "weight_lora", "full_ft", "adapters", "bitfit"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=15000)
    p.add_argument("--eval_frequency", type=int, default=500)
    return p.parse_args()


def create_shared_instruction_encoder(config):
    lora_config = config.get("lora", {})
    model_config = config.get("model", {})
    instruction_encoder = InstructionEncoder(
        vocab_size=model_config.get("vocab_size", 30522),
        hidden_dim=model_config.get("hidden_dim", 768),
        output_dim=lora_config.get("instruction_encoder_hidden", 256),
        num_layers=lora_config.get("instruction_encoder_layers", 2),
    )
    instruction_to_hidden = torch.nn.Linear(
        lora_config.get("instruction_encoder_hidden", 256),
        model_config.get("hidden_dim", 768),
    )
    torch.nn.init.zeros_(instruction_to_hidden.weight)
    torch.nn.init.zeros_(instruction_to_hidden.bias)
    return instruction_encoder, instruction_to_hidden


def setup_model(config, method_name):
    base_model = MaskedDiffusionTransformer(config)
    if config.get("method", {}).get("freeze_base", True) and method_name != "full_ft":
        for p in base_model.parameters():
            p.requires_grad = False
    lora_module = None
    instruction_encoder = None
    instruction_to_hidden = None
    if method_name == "lora_diffusion":
        lora_module = LoRADiffusionModule(config)
        encoder_frozen = config.get("instruction_encoder_frozen", True)
        if encoder_frozen:
            for p in lora_module.instruction_encoder.parameters():
                p.requires_grad = False
            for p in lora_module.instruction_to_hidden.parameters():
                p.requires_grad = False
    elif method_name == "weight_lora":
        method_lora = config.get("method", {}).get("lora", {})
        lora_config = {**config.get("lora", {}), **method_lora}
        apply_weight_lora_to_model(
            base_model,
            rank=lora_config.get("rank", 64),
            alpha=lora_config.get("alpha", 16),
            target_modules=lora_config.get("target_modules", ["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"]),
        )
    elif method_name == "adapters":
        adapter_config = {**config.get("adapter", {}), **config.get("method", {}).get("adapter", {})}
        apply_adapters_to_model(
            base_model,
            bottleneck_dim=adapter_config.get("bottleneck_dim", 256),
            insert_after=adapter_config.get("insert_after", ["attention", "ffn"]),
        )
    elif method_name == "bitfit":
        bitfit_config = {**config.get("bitfit", {}), **config.get("method", {}).get("bitfit", {})}
        setup_bitfit(base_model, train_layer_norm=bitfit_config.get("train_layer_norm", True))
    if lora_module is None and method_name in ("full_ft", "weight_lora", "adapters", "bitfit"):
        instruction_encoder, instruction_to_hidden = create_shared_instruction_encoder(config)
        encoder_frozen = config.get("instruction_encoder_frozen", True)
        if encoder_frozen:
            for p in instruction_encoder.parameters():
                p.requires_grad = False
            for p in instruction_to_hidden.parameters():
                p.requires_grad = False
    return base_model, lora_module, instruction_encoder, instruction_to_hidden


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path, task_name=TASK_NAMES[0], method_name=args.method)
    config["training"]["max_steps"] = args.max_steps
    config["training"]["eval_frequency"] = args.eval_frequency
    config["training"]["seed"] = args.seed
    config["output"]["base_dir"] = args.output_dir
    config["output"]["checkpoint_dir"] = args.output_dir
    config["task"] = config.get("task", {})
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs", experiment_name="multitask_joint_" + args.method)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer", {}).get("name", "bert-base-uncased"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    data_config = config["data"]
    cache_dir = data_config["cache_dir"]
    max_seq = config["model"]["max_seq_length"]
    max_train = data_config.get("max_train_samples")
    max_eval = data_config.get("max_eval_samples")
    train_datasets = []
    eval_datasets = []
    for idx, task_name in enumerate(TASK_NAMES):
        task_cfg = load_config(config_path, task_name=task_name, method_name=args.method)
        tc = task_cfg.get("task", {})
        tc["task_id"] = idx
        tr = get_task_loader(task_name, "train", cache_dir, tc, tokenizer, max_train, max_seq)
        ev = get_task_loader(task_name, "validation", cache_dir, tc, tokenizer, max_eval, max_seq)
        train_datasets.append(tr)
        eval_datasets.append(ev)
        logger.info("Task %s (id=%d): train=%d val=%d", task_name, idx, len(tr), len(ev))
    train_dataset = ConcatDataset(train_datasets)
    eval_dataset = ConcatDataset(eval_datasets)
    collator = DiffusionCollator(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collator, num_workers=data_config.get("num_workers", 4), pin_memory=data_config.get("pin_memory", True))
    eval_loader = DataLoader(eval_dataset, batch_size=config["training"]["eval_batch_size"], shuffle=False, collate_fn=collator, num_workers=data_config.get("num_workers", 4), pin_memory=data_config.get("pin_memory", True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, lora_module, instruction_encoder, instruction_to_hidden = setup_model(config, args.method)
    params = list(model.parameters()) if lora_module is None else list(lora_module.parameters())
    if instruction_encoder is not None:
        params += [p for p in instruction_encoder.parameters() if p.requires_grad]
    if instruction_to_hidden is not None:
        params += [p for p in instruction_to_hidden.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"], betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]), eps=config["training"]["adam_epsilon"])
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps - config["training"].get("warmup_steps", 500))
    trainer = DiffusionTrainer(model=model, train_dataloader=train_loader, eval_dataloader=eval_loader, optimizer=optimizer, scheduler=scheduler, config=config, lora_module=lora_module, device=device, instruction_encoder=instruction_encoder, instruction_to_hidden=instruction_to_hidden)
    trainer.train()
    run_info = {"seed": args.seed, "task": "multitask_joint", "tasks": TASK_NAMES, "method": args.method, "command": " ".join([sys.executable] + sys.argv)}
    try:
        import subprocess
        run_info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        run_info["git_commit"] = None
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    logger.info("Joint multi-task training complete. Run evaluate.py per task (sst2, qnli, mrpc) for per-task metrics.")


if __name__ == "__main__":
    main()
