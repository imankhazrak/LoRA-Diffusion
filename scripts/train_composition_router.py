#!/usr/bin/env python3
"""
Train only the composition router for multi-task LoRA-Diffusion.

Loads pre-trained single-task LoRA modules (e.g. SST-2, MRPC, QNLI), freezes them,
and trains the router to predict task_id from instruction embedding (cross-entropy).
Saves router.pt to a dedicated directory (e.g. outputs/composition_router/).
"""

import argparse
import logging
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MultiTaskLoRAComposer
from src.data import get_task_loader, DiffusionCollator
from src.utils import load_config, setup_logging, set_seed

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train composition router only")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base config",
    )
    parser.add_argument(
        "--task_checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to single-task checkpoint dirs (order: sst2, mrpc, qnli)",
    )
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="+",
        default=["sst2", "mrpc", "qnli"],
        help="Task names corresponding to task_checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/composition_router",
        help="Directory to save router.pt",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Router training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for router",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    config_path = Path(args.config)
    config = load_config(config_path, task_name=args.task_names[0], method_name="lora_diffusion")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs", experiment_name="composition_router")

    if len(args.task_checkpoints) != len(args.task_names):
        raise ValueError(
            f"task_checkpoints and task_names must have same length: "
            f"{len(args.task_checkpoints)} vs {len(args.task_names)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    tokenizer_name = config.get("tokenizer", {}).get("name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    cache_dir = config["data"]["cache_dir"]
    max_seq_length = config["model"]["max_seq_length"]
    max_train = config["data"].get("max_train_samples")

    # Mixed dataset: one dataset per task with task_id set
    datasets = []
    for idx, (task_name, ckpt_path) in enumerate(zip(args.task_names, args.task_checkpoints)):
        task_config = load_config(config_path, task_name=task_name, method_name="lora_diffusion")
        task_config = {**task_config.get("task", {}), "task_id": idx}
        ds = get_task_loader(
            task_name=task_name,
            split="train",
            cache_dir=cache_dir,
            task_config=task_config,
            tokenizer=tokenizer,
            max_samples=max_train,
            max_seq_length=max_seq_length,
        )
        datasets.append(ds)
        logger.info(f"Task {task_name} (id={idx}): {len(ds)} train examples")

    mixed = ConcatDataset(datasets)
    collator = DiffusionCollator(tokenizer=tokenizer)
    train_loader = DataLoader(
        mixed,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    # Composer with frozen adapters; we will train only the router
    composer = MultiTaskLoRAComposer(
        config=config,
        task_names=args.task_names,
    ).to(device)

    for task_name, ckpt_path in zip(args.task_names, args.task_checkpoints):
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        composer.load_task_module(task_name, p)
    logger.info("Loaded all task LoRA modules (frozen)")

    # Freeze instruction encoder so we only train the router
    for p in composer.instruction_encoder.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in composer.router.parameters() if p.requires_grad)
    logger.info(f"Router trainable parameters: {trainable}")

    optimizer = torch.optim.AdamW(composer.router.parameters(), lr=args.lr)
    composer.train()
    step = 0
    epoch = 0
    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            instruction_ids = batch["instruction_ids"].to(device)
            instruction_mask = batch["instruction_mask"].to(device)
            task_labels = batch["task_labels"].to(device)

            instruction_emb = composer.instruction_encoder(
                instruction_ids,
                attention_mask=instruction_mask,
            )
            loss = composer.compute_router_loss(instruction_emb, task_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                logger.info(f"Step {step}/{args.max_steps} loss={loss.item():.4f}")
            if step >= args.max_steps:
                break

    router_path = output_dir / "router.pt"
    torch.save(composer.router.state_dict(), router_path)
    logger.info(f"Saved router to {router_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
