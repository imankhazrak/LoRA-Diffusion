"""Checkpoint saving and loading utilities."""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    lora_module: Optional[torch.nn.Module] = None,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
):
    """
    Save a training checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        lora_module: Optional LoRA module
        step: Current training step
        metrics: Optional metrics to save
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = checkpoint_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save LoRA module if present
    if lora_module is not None:
        lora_path = checkpoint_dir / "lora_module.pt"
        torch.save(lora_module.state_dict(), lora_path)
        logger.info(f"Saved LoRA module to {lora_path}")
        
        # Calculate LoRA module size
        lora_size_mb = lora_path.stat().st_size / (1024 * 1024)
        logger.info(f"LoRA module size: {lora_size_mb:.2f} MB")
    
    # Save optimizer and scheduler
    if optimizer is not None:
        training_state = {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        
        if scheduler is not None:
            training_state["scheduler_state_dict"] = scheduler.state_dict()
        
        if metrics is not None:
            training_state["metrics"] = metrics
        
        state_path = checkpoint_dir / "training_state.pt"
        torch.save(training_state, state_path)
        logger.info(f"Saved training state to {state_path}")


def load_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    lora_module: Optional[torch.nn.Module] = None,
    device: str = "cuda",
) -> int:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        lora_module: Optional LoRA module to load state into
        device: Device to load tensors to
        
    Returns:
        Training step from checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Load model
    model_path = checkpoint_dir / "model.pt"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"Model checkpoint not found: {model_path}")
    
    # Load LoRA module
    if lora_module is not None:
        lora_path = checkpoint_dir / "lora_module.pt"
        if lora_path.exists():
            state_dict = torch.load(lora_path, map_location=device)
            lora_module.load_state_dict(state_dict)
            logger.info(f"Loaded LoRA module from {lora_path}")
        else:
            logger.warning(f"LoRA checkpoint not found: {lora_path}")
    
    # Load training state
    step = 0
    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        training_state = torch.load(state_path, map_location=device)
        step = training_state.get("step", 0)
        
        if optimizer is not None and "optimizer_state_dict" in training_state:
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            logger.info("Loaded optimizer state")
        
        if scheduler is not None and "scheduler_state_dict" in training_state:
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
            logger.info("Loaded scheduler state")
        
        if "metrics" in training_state:
            logger.info(f"Checkpoint metrics: {training_state['metrics']}")
    else:
        logger.warning(f"Training state not found: {state_path}")
    
    return step
