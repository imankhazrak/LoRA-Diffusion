#!/usr/bin/env python3
"""Script for zero-shot task composition using multiple trained LoRA modules."""

import argparse
import logging
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MaskedDiffusionTransformer, MultiTaskLoRAComposer
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compose multiple LoRA modules for zero-shot task combination")
    
    parser.add_argument(
        "--task_modules",
        type=str,
        nargs="+",
        required=True,
        help="Paths to checkpoint directories containing trained LoRA modules",
    )
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="+",
        required=True,
        help="Names of tasks corresponding to task_modules (e.g., sst2 squad xsum)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Instruction text describing the composite task",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for generated text",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length for generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(Path("./logs"), experiment_name="composition")
    
    logger.info("=" * 80)
    logger.info("Zero-Shot Task Composition")
    logger.info("=" * 80)
    logger.info(f"Task modules: {args.task_modules}")
    logger.info(f"Task names: {args.task_names}")
    logger.info(f"Instruction: {args.instruction}")
    
    if len(args.task_modules) != len(args.task_names):
        raise ValueError(
            f"Number of task_modules ({len(args.task_modules)}) must match "
            f"number of task_names ({len(args.task_names)})"
        )
    
    # Load configuration
    config = load_config(Path(args.config))
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_name = config.get("tokenizer", {}).get("name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    logger.info(f"Loaded tokenizer: {tokenizer_name}")
    
    # Load base model
    logger.info("Loading base diffusion model...")
    model = MaskedDiffusionTransformer(config).to(device)
    model.eval()
    
    # Create multi-task composer
    logger.info("Creating multi-task LoRA composer...")
    composer = MultiTaskLoRAComposer(
        config=config,
        task_names=args.task_names,
    ).to(device)
    
    # Load task-specific LoRA modules
    logger.info("Loading task-specific LoRA modules...")
    for task_name, checkpoint_path in zip(args.task_names, args.task_modules):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading LoRA module for {task_name} from {checkpoint_path}")
        composer.load_task_module(task_name, checkpoint_path)
    
    # Load router if available (optional, for inspection)
    router_path = Path(args.task_modules[0]).parent / "router.pt"
    if router_path.exists():
        logger.info(f"Loading router from {router_path}")
        router_state = torch.load(router_path, map_location=device)
        composer.router.load_state_dict(router_state)
    
    composer.eval()
    
    # Tokenize instruction
    instruction_encoding = tokenizer(
        args.instruction,
        max_length=512,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    instruction_ids = instruction_encoding["input_ids"].to(device)
    instruction_mask = instruction_encoding["attention_mask"].to(device)
    
    # Expand to batch size
    if args.batch_size > 1:
        instruction_ids = instruction_ids.repeat(args.batch_size, 1)
        instruction_mask = instruction_mask.repeat(args.batch_size, 1)
    
    # Get task weights (for inspection)
    with torch.no_grad():
        task_weights = composer.get_task_weights(instruction_ids, instruction_mask)
        logger.info("Task weights:")
        for task_name, weight in zip(args.task_names, task_weights[0]):
            logger.info(f"  {task_name}: {weight:.4f}")
    
    # Generate with composition
    logger.info("Generating with task composition...")
    with torch.no_grad():
        generated_ids = model.sample_with_composition(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            instruction_ids=instruction_ids,
            instruction_mask=instruction_mask,
            composer=composer,
            device=device,
        )
    
    # Decode generated text
    generated_texts = []
    for ids in generated_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        generated_texts.append(text)
        logger.info(f"Generated: {text}")
    
    # Save results
    results = {
        "instruction": args.instruction,
        "task_modules": args.task_modules,
        "task_names": args.task_names,
        "task_weights": task_weights[0].cpu().tolist(),
        "generated_texts": generated_texts,
    }
    
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")
    else:
        # Print to stdout
        print(json.dumps(results, indent=2))
    
    logger.info("Composition complete!")


if __name__ == "__main__":
    main()
