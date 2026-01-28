#!/usr/bin/env python3
"""
Comprehensive example demonstrating LoRA-Diffusion usage.

This script shows:
1. Loading and configuring the model
2. Training with LoRA-Diffusion
3. Evaluating the model
4. Comparing with baselines
5. Analyzing results
"""

import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MaskedDiffusionTransformer, LoRADiffusionModule
from src.data import get_task_loader, DiffusionCollator
from src.training import DiffusionTrainer
from src.utils import load_config, set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_training():
    """Example 1: Basic LoRA-Diffusion training."""
    logger.info("=" * 80)
    logger.info("Example 1: Basic LoRA-Diffusion Training")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(
        Path("configs/base_config.yaml"),
        task_name="sst2",
        method_name="lora_diffusion",
    )
    
    # Override for quick example
    config["training"]["max_steps"] = 100
    config["data"]["max_train_samples"] = 1000
    config["data"]["max_eval_samples"] = 200
    
    # Set seed
    set_seed(42)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load data
    train_dataset = get_task_loader(
        task_name="sst2",
        split="train",
        cache_dir="./data/cache",
        task_config=config["task"],
        tokenizer=tokenizer,
        max_samples=1000,
        max_seq_length=128,
    )
    
    eval_dataset = get_task_loader(
        task_name="sst2",
        split="validation",
        cache_dir="./data/cache",
        task_config=config["task"],
        tokenizer=tokenizer,
        max_samples=200,
        max_seq_length=128,
    )
    
    # Create dataloaders
    collator = DiffusionCollator(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
    )
    
    # Create models
    base_model = MaskedDiffusionTransformer(config).to(device)
    
    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Create LoRA module
    lora_module = LoRADiffusionModule(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    lora_params = sum(p.numel() for p in lora_module.parameters())
    logger.info(f"Base model parameters: {total_params:,}")
    logger.info(f"LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}%)")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        lora_module.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )
    
    # Setup scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=base_model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        lora_module=lora_module,
        device=device,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete!")
    logger.info(f"Checkpoint saved to: {config['output']['checkpoint_dir']}")


def example_method_comparison():
    """Example 2: Compare different PEFT methods."""
    logger.info("=" * 80)
    logger.info("Example 2: Method Comparison")
    logger.info("=" * 80)
    
    methods = ["lora_diffusion", "weight_lora", "adapters", "bitfit"]
    results = {}
    
    for method in methods:
        logger.info(f"\nTraining with {method}...")
        
        # Load config
        config = load_config(
            Path("configs/base_config.yaml"),
            task_name="sst2",
            method_name=method,
        )
        
        # Quick training for demo
        config["training"]["max_steps"] = 50
        config["data"]["max_train_samples"] = 500
        
        # ... (setup and train as in example 1)
        
        # Store results
        results[method] = {
            "final_loss": 0.0,  # Would get from trainer
            "accuracy": 0.0,    # Would get from evaluation
        }
    
    # Print comparison
    logger.info("\n" + "=" * 80)
    logger.info("Method Comparison Results")
    logger.info("=" * 80)
    for method, metrics in results.items():
        logger.info(f"{method}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")


def example_step_adaptive_ranks():
    """Example 3: Demonstrate step-adaptive rank allocation."""
    logger.info("=" * 80)
    logger.info("Example 3: Step-Adaptive Rank Analysis")
    logger.info("=" * 80)
    
    # Create LoRA module
    config = load_config(
        Path("configs/base_config.yaml"),
        task_name="sst2",
        method_name="lora_diffusion",
    )
    
    lora_module = LoRADiffusionModule(config)
    
    # Show rank allocation across diffusion steps
    T = config["diffusion"]["num_steps"]
    logger.info(f"\nRank allocation across {T} diffusion steps:")
    logger.info("-" * 60)
    
    test_steps = [5, 20, 35, 50, 65, 80, 95]
    for t in test_steps:
        adapters, scaling = lora_module.get_step_config(t)
        
        if adapters == lora_module.adapters_early:
            phase = "Early"
            rank = config["lora"]["rank_early"]
        elif adapters == lora_module.adapters_mid:
            phase = "Mid"
            rank = config["lora"]["rank_mid"]
        else:
            phase = "Late"
            rank = config["lora"]["rank_late"]
        
        logger.info(f"Step {t:3d}: Phase={phase:5s}, Rank={rank:2d}, Scaling={scaling:.2f}")
    
    # Show parameter breakdown
    param_counts = lora_module.count_parameters()
    logger.info("\n" + "=" * 60)
    logger.info("Parameter Breakdown:")
    logger.info("=" * 60)
    for key, count in param_counts.items():
        logger.info(f"{key:20s}: {count:10,} parameters")


def example_trajectory_perturbation():
    """Example 4: Visualize trajectory perturbations."""
    logger.info("=" * 80)
    logger.info("Example 4: Trajectory Perturbation Visualization")
    logger.info("=" * 80)
    
    # Create models
    config = load_config(
        Path("configs/base_config.yaml"),
        task_name="sst2",
        method_name="lora_diffusion",
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_model = MaskedDiffusionTransformer(config).to(device)
    lora_module = LoRADiffusionModule(config).to(device)
    
    # Create dummy input
    batch_size = 4
    seq_len = 16
    hidden_dim = config["model"]["hidden_dim"]
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)
    timesteps = torch.tensor([80, 50, 30, 10]).to(device)  # Different phases
    instruction_ids = torch.randint(0, 1000, (batch_size, 10)).to(device)
    instruction_mask = torch.ones(batch_size, 10).to(device)
    
    # Compute perturbations
    with torch.no_grad():
        delta = lora_module(
            hidden_states=hidden_states,
            timesteps=timesteps,
            instruction_ids=instruction_ids,
            instruction_mask=instruction_mask,
        )
    
    # Analyze perturbation magnitudes
    logger.info("\nPerturbation magnitudes by timestep:")
    logger.info("-" * 60)
    for i, t in enumerate(timesteps):
        magnitude = delta[i].norm(dim=-1).mean().item()
        logger.info(f"Timestep {t.item():3d}: Magnitude = {magnitude:.4f}")
    
    logger.info("\nNote: Early steps should have larger perturbations (higher scaling)")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LoRA-Diffusion Comprehensive Examples")
    print("=" * 80 + "\n")
    
    try:
        # Run examples
        example_basic_training()
        print("\n")
        
        # Uncomment to run additional examples:
        # example_method_comparison()
        # example_step_adaptive_ranks()
        # example_trajectory_perturbation()
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        raise
    
    print("\n" + "=" * 80)
    print("All examples complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
