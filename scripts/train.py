#!/usr/bin/env python3
"""Main training script for LoRA-Diffusion."""

import argparse
import logging
from pathlib import Path
import sys
import os

# Set cache directories BEFORE importing any libraries that might use them
# This prevents read-only filesystem errors
# Force override even if set to /data (read-only)
cache_base = Path(__file__).parent.parent / "data"
cache_base.mkdir(parents=True, exist_ok=True)

# Force set cache directories (override any existing /data paths)
os.environ["HF_HOME"] = str(cache_base / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(cache_base / "transformers_cache")
os.environ["HF_DATASETS_CACHE"] = str(cache_base / "datasets_cache")
os.environ["TORCH_HOME"] = str(cache_base / "torch_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_base / "hub_cache")

# Create all cache directories
for cache_name in ["hf_cache", "transformers_cache", "datasets_cache", "torch_cache", "hub_cache"]:
    cache_dir = cache_base / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MaskedDiffusionTransformer, LoRADiffusionModule
from src.models.baselines import (
    apply_weight_lora_to_model,
    setup_bitfit,
    apply_adapters_to_model,
)
from src.data import get_task_loader, DiffusionCollator
from src.training import DiffusionTrainer
from src.utils import load_config, setup_logging, set_seed, parse_override_string, apply_override

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LoRA-Diffusion")
    
    # Config files
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (sst2, squad, xsum, etc.)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lora_diffusion",
        choices=["lora_diffusion", "weight_lora", "full_ft", "prefix_tuning", "adapters", "bitfit"],
        help="Training method",
    )
    
    # Override options
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Use subset of data (for quick testing)",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=None,
        help="Evaluation frequency",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    # Additional overrides (format: key.subkey=value)
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=[],
        help="Config overrides in format key.subkey=value",
    )
    
    return parser.parse_args()


def setup_model(config, method_name):
    """Initialize model based on method."""
    logger.info(f"Initializing model with method: {method_name}")
    
    # Create base diffusion model
    base_model = MaskedDiffusionTransformer(config)
    logger.info(f"Created base diffusion model")
    
    # Freeze base model for PEFT methods
    if config.get("method", {}).get("freeze_base", True) and method_name != "full_ft":
        for param in base_model.parameters():
            param.requires_grad = False
        logger.info("Froze base model parameters")
    
    lora_module = None
    
    if method_name == "lora_diffusion":
        # Create LoRA-Diffusion module
        lora_module = LoRADiffusionModule(config)
        logger.info("Created LoRA-Diffusion module")
        
        # Log parameter counts
        lora_counts = lora_module.count_parameters()
        logger.info(f"LoRA parameters: {lora_counts}")
        
    elif method_name == "weight_lora":
        # Apply weight LoRA to model (use method-specific config so eval loads same structure)
        method_lora = config.get("method", {}).get("lora", {})
        lora_config = {**config.get("lora", {}), **method_lora}
        rank = lora_config.get("rank", 64)
        alpha = lora_config.get("alpha", 16)
        target_modules = lora_config.get(
            "target_modules",
            ["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"],
        )
        
        count = apply_weight_lora_to_model(
            base_model,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
        )
        logger.info(f"Applied weight LoRA to {count} modules")
        
    elif method_name == "adapters":
        # Add adapter modules (use method-specific config so eval loads same structure)
        method_adapter = config.get("method", {}).get("adapter", {})
        adapter_config = {**config.get("adapter", {}), **method_adapter}
        bottleneck_dim = adapter_config.get("bottleneck_dim", 256)
        insert_after = adapter_config.get("insert_after", ["attention", "ffn"])
        
        count = apply_adapters_to_model(
            base_model,
            bottleneck_dim=bottleneck_dim,
            insert_after=insert_after,
        )
        logger.info(f"Added {count} adapter modules")
        
    elif method_name == "bitfit":
        # Setup BitFit (use method-specific config so eval matches)
        method_bitfit = config.get("method", {}).get("bitfit", {})
        bitfit_config = {**config.get("bitfit", {}), **method_bitfit}
        train_layer_norm = bitfit_config.get("train_layer_norm", True)
        
        count = setup_bitfit(base_model, train_layer_norm=train_layer_norm)
        logger.info(f"BitFit: {count} trainable parameters")
        
    elif method_name == "prefix_tuning":
        # Create prefix tuning module
        from src.models.baselines import PrefixTuningModule
        prefix_config = config.get("prefix", {})
        num_virtual_tokens = prefix_config.get("num_virtual_tokens", 32)
        prefix_hidden_dim = prefix_config.get("prefix_hidden_dim", 512)
        reparameterization = prefix_config.get("reparameterization", True)
        
        prefix_module = PrefixTuningModule(
            num_layers=base_model.num_layers,
            num_virtual_tokens=num_virtual_tokens,
            hidden_dim=base_model.hidden_dim,
            num_heads=base_model.num_heads,
            prefix_hidden_dim=prefix_hidden_dim,
            reparameterization=reparameterization,
            dropout=prefix_config.get("prefix_dropout", 0.1),
        )
        logger.info(f"Created prefix tuning module with {num_virtual_tokens} virtual tokens")
        
        # Store prefix module for use in training
        base_model.prefix_module = prefix_module
        lora_module = prefix_module  # Use same variable name for compatibility
        
        # Log parameter count
        prefix_params = sum(p.numel() for p in prefix_module.parameters() if p.requires_grad)
        logger.info(f"Prefix tuning parameters: {prefix_params:,}")
        
    elif method_name == "full_ft":
        logger.info("Using full fine-tuning")
    
    return base_model, lora_module


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(
        Path(args.config),
        task_name=args.task,
        method_name=args.method,
    )
    
    # Apply command-line overrides
    overrides = {}
    if args.output_dir:
        overrides["output"] = overrides.get("output", {})
        overrides["output"]["base_dir"] = args.output_dir
        # Save checkpoints inside this run's output dir so each run has its own final_model/
        overrides["output"]["checkpoint_dir"] = args.output_dir
    if args.seed:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["seed"] = args.seed
    if args.max_steps:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["max_steps"] = args.max_steps
    if args.eval_frequency:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["eval_frequency"] = args.eval_frequency
    if args.subset_size:
        overrides["data"] = overrides.get("data", {})
        overrides["data"]["max_train_samples"] = args.subset_size
        overrides["data"]["max_eval_samples"] = min(args.subset_size // 10, 1000)
    
    # Parse additional overrides
    for override_str in args.overrides:
        key_path, value = parse_override_string(override_str)
        apply_override(overrides, key_path, value)
    
    # Merge overrides into config
    if overrides:
        from src.utils.config import merge_configs
        config = merge_configs(config, overrides)
    
    # Setup logging
    log_dir = Path(config["output"]["log_dir"])
    experiment_name = f"{args.task}_{args.method}"
    setup_logging(log_dir, experiment_name=experiment_name)
    
    logger.info("=" * 80)
    logger.info(f"Training {args.method} on {args.task}")
    logger.info("=" * 80)
    
    # Set random seed
    seed = config["training"]["seed"]
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_name = config.get("tokenizer", {}).get("name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    logger.info(f"Loaded tokenizer: {tokenizer_name}")
    
    # Load data
    logger.info("Loading datasets...")
    task_config = config.get("task", {})
    data_config = config["data"]
    
    train_dataset = get_task_loader(
        task_name=args.task,
        split="train",
        cache_dir=data_config["cache_dir"],
        task_config=task_config,
        tokenizer=tokenizer,
        max_samples=data_config.get("max_train_samples"),
        max_seq_length=config["model"]["max_seq_length"],
    )
    
    eval_dataset = get_task_loader(
        task_name=args.task,
        split="validation",
        cache_dir=data_config["cache_dir"],
        task_config=task_config,
        tokenizer=tokenizer,
        max_samples=data_config.get("max_eval_samples"),
        max_seq_length=config["model"]["max_seq_length"],
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Create data loaders
    collator = DiffusionCollator(tokenizer=tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
    )
    
    # Setup model
    model, lora_module = setup_model(config, args.method)
    
    # Setup optimizer
    logger.info("Setting up optimizer...")
    if lora_module is not None:
        params_to_optimize = list(lora_module.parameters())
    else:
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        eps=config["training"]["adam_epsilon"],
    )
    
    # Setup scheduler
    num_training_steps = config["training"]["max_steps"]
    num_warmup_steps = config["training"]["warmup_steps"]
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps)
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        lora_module=lora_module,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(Path(args.resume_from))
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("interrupted")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
