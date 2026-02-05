"""Main trainer class for LoRA-Diffusion training."""

import logging
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# Prefer torch.amp.autocast (PyTorch 2.0+) to avoid deprecation warning
if hasattr(torch.amp, "autocast"):
    _autocast = lambda **kw: torch.amp.autocast("cuda", **kw)
else:
    from torch.cuda.amp import autocast as _autocast
import json

from .losses import compute_diffusion_loss

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """Trainer for diffusion models with LoRA-Diffusion support."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: Dict[str, Any],
        lora_module: Optional[nn.Module] = None,
        router: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: Base diffusion model
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Full configuration
            lora_module: Optional LoRA module
            router: Optional task router for multi-task composition
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.lora_module = lora_module.to(device) if lora_module else None
        self.router = router.to(device) if router else None
        self.device = device
        
        # Training config
        train_config = config["training"]
        self.max_steps = train_config["max_steps"]
        self.gradient_accumulation_steps = train_config["gradient_accumulation_steps"]
        self.max_grad_norm = train_config["max_grad_norm"]
        self.logging_steps = train_config["logging_steps"]
        self.eval_frequency = train_config["eval_frequency"]
        self.save_frequency = train_config["save_frequency"]
        
        # Mixed precision
        self.use_amp = train_config["mixed_precision"] in ["fp16", "bf16"]
        self.scaler = GradScaler() if train_config["mixed_precision"] == "fp16" else None
        self.amp_dtype = torch.float16 if train_config["mixed_precision"] == "fp16" else torch.bfloat16
        
        # Output paths
        output_config = config["output"]
        self.output_dir = Path(output_config["base_dir"])
        self.checkpoint_dir = Path(output_config["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.global_step = 0
        self.best_metric = float("-inf")
        self.training_history = []
        self.evaluation_history = []  # Track evaluation metrics
        
        # Count parameters
        self._log_parameter_counts()
    
    def _log_parameter_counts(self):
        """Log parameter counts."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.lora_module:
            lora_params = sum(p.numel() for p in self.lora_module.parameters())
            trainable_params += lora_params
            logger.info(f"LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}%)")
        
        if self.router:
            router_params = sum(p.numel() for p in self.router.parameters())
            trainable_params += router_params
            logger.info(f"Router parameters: {router_params:,} ({router_params/total_params*100:.2f}%)")
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        self.model.train()
        if self.lora_module:
            self.lora_module.train()
        if self.router:
            self.router.train()
        
        train_iterator = iter(self.train_dataloader)
        accumulated_loss = 0.0
        accumulated_metrics = {}
        step_start_time = time.time()
        
        pbar = tqdm(total=self.max_steps, desc="Training")
        
        while self.global_step < self.max_steps:
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
            
            # Forward pass
            # Extract task labels if present
            task_labels = batch.get("task_labels", None)
            
            with _autocast(enabled=self.use_amp, dtype=self.amp_dtype if self.use_amp else torch.float32):
                loss, metrics = compute_diffusion_loss(
                    model=self.model,
                    batch=batch,
                    lora_module=self.lora_module,
                    config=self.config,
                    router=self.router,
                    task_labels=task_labels,
                )
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate metrics (per-batch average, same as training log)
            accumulated_loss += loss.item()
            for key, value in metrics.items():
                accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value
            
            # Update step
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    
                    params_to_clip = []
                    if self.lora_module:
                        params_to_clip.extend(list(self.lora_module.parameters()))
                    if self.router:
                        params_to_clip.extend(list(self.router.parameters()))
                    if not params_to_clip:
                        params_to_clip = [p for p in self.model.parameters() if p.requires_grad]
                    
                    torch.nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Log
                if (self.global_step + 1) % self.logging_steps == 0:
                    step_time = time.time() - step_start_time
                    lr = self.scheduler.get_last_lr()[0]
                    avg_metrics = {k: v / self.logging_steps for k, v in accumulated_metrics.items()}
                    
                    log_str = f"Step {self.global_step + 1}/{self.max_steps} | "
                    log_str += f"Loss: {avg_metrics['loss']:.4f} | "
                    log_str += f"Acc: {avg_metrics['accuracy']:.4f} | "
                    log_str += f"LR: {lr:.2e} | "
                    log_str += f"Time: {step_time/self.logging_steps:.2f}s/step"
                    
                    logger.info(log_str)
                    pbar.set_postfix(avg_metrics)
                    
                    # Record history
                    self.training_history.append({
                        "step": self.global_step + 1,
                        "metrics": avg_metrics,
                        "lr": lr,
                        "time_per_step": step_time / self.logging_steps,
                    })
                    
                    accumulated_loss = 0.0
                    accumulated_metrics = {}
                    step_start_time = time.time()
                
                # Evaluate
                if (self.global_step + 1) % self.eval_frequency == 0:
                    eval_metrics = self.evaluate()
                    logger.info(f"Eval metrics: {eval_metrics}")
                    
                    # Record evaluation metrics
                    self.evaluation_history.append({
                        "step": self.global_step + 1,
                        "metrics": eval_metrics,
                    })
                    
                    # Save evaluation metrics to file immediately
                    eval_history_path = self.output_dir / "evaluation_history.json"
                    with open(eval_history_path, "w") as f:
                        json.dump(self.evaluation_history, f, indent=2)
                    
                    # Save best model
                    # Get primary metric from config (task configs have metrics.primary)
                    metrics_config = self.config.get("metrics", {})
                    primary_metric_name = metrics_config.get("primary", "loss")
                    primary_metric = eval_metrics.get(
                        primary_metric_name,
                        eval_metrics.get("loss", 0.0)
                    )
                    if primary_metric > self.best_metric:
                        self.best_metric = primary_metric
                        self.save_checkpoint(f"best_model")
                        logger.info(f"New best model! Metric: {primary_metric:.4f}")
                    
                    self.model.train()
                    if self.lora_module:
                        self.lora_module.train()
                    if self.router:
                        self.router.train()
                
                # Save checkpoint
                if (self.global_step + 1) % self.save_frequency == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step + 1}")
            
            self.global_step += 1
            pbar.update(1)
        
        pbar.close()
        logger.info("Training complete!")
        
        # Save final checkpoint
        self.save_checkpoint("final_model")
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Save evaluation history (final save)
        eval_history_path = self.output_dir / "evaluation_history.json"
        with open(eval_history_path, "w") as f:
            json.dump(self.evaluation_history, f, indent=2)
        logger.info(f"Saved evaluation history to {eval_history_path}")
        
        # Save final summary
        summary = {
            "total_steps": self.global_step,
            "best_metric": self.best_metric,
            "final_training_metrics": self.training_history[-1]["metrics"] if self.training_history else {},
            "final_eval_metrics": self.evaluation_history[-1]["metrics"] if self.evaluation_history else {},
            "config": self.config,
        }
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved training summary to {summary_path}")
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on eval set."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        if self.lora_module:
            self.lora_module.eval()
        if self.router:
            self.router.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_metrics = {}
        
        # For task-specific metrics (e.g., classification accuracy), generate text
        task_config = self.config.get("task", {})
        task_type = task_config.get("type", None)
        compute_task_metrics = task_type in ["classification", "qa", "summarization"]
        
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            task_labels = batch.get("task_labels", None)
            
            with _autocast(enabled=self.use_amp, dtype=self.amp_dtype if self.use_amp else torch.float32):
                loss, metrics = compute_diffusion_loss(
                    model=self.model,
                    batch=batch,
                    lora_module=self.lora_module,
                    config=self.config,
                    router=self.router,
                    task_labels=task_labels,
                )
            
            batch_size = batch["target_ids"].size(0)
            total_loss += loss.item()
            num_batches += 1
            for key, value in metrics.items():
                all_metrics[key] = all_metrics.get(key, 0.0) + value
            
            # Generate text for task-specific metrics
            if compute_task_metrics:
                target_texts = batch.get("target_texts", [])
                all_references.extend(target_texts)
                
                # Generate predictions
                instruction_ids = batch["instruction_ids"].to(self.device)
                instruction_mask = batch["instruction_mask"].to(self.device)
                
                # For classification tasks, use fixed short sequence length (3-5 tokens)
                # For other tasks, use the target sequence length
                if task_type == "classification":
                    seq_len = 5  # Fixed short length for classification
                else:
                    seq_len = batch["target_ids"].size(1)
                
                if self.lora_module:
                    instruction_emb = self.lora_module.instruction_encoder(
                        instruction_ids,
                        attention_mask=instruction_mask,
                    )
                    # Project to hidden_dim for base model conditioning
                    instruction_emb_for_base = self.lora_module.instruction_to_hidden(instruction_emb)
                else:
                    instruction_emb_for_base = None
                
                # Get tokenizer and label names for classification
                tokenizer = self.eval_dataloader.dataset.tokenizer
                label_names = task_config.get("label_names", None)
                
                # Use classification-specific sampling if available
                if task_type == "classification" and hasattr(self.model, 'sample_classification'):
                    samples = self.model.sample_classification(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        instruction_embedding=instruction_emb_for_base,
                        device=self.device,
                        tokenizer=tokenizer,
                        label_names=label_names,
                        early_stop=True,
                    )
                else:
                    # Use standard sampling with greedy option for classification
                    samples = self.model.sample(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        instruction_embedding=instruction_emb_for_base,
                        device=self.device,
                        greedy=(task_type == "classification"),  # Use greedy for classification
                    )
                
                # Decode predictions
                for sample in samples:
                    pred_text = tokenizer.decode(sample, skip_special_tokens=True).strip()
                    all_predictions.append(pred_text)
        
        # Average token-level metrics (per-batch average, same as training log)
        n = max(num_batches, 1)
        eval_metrics = {
            key: value / n
            for key, value in all_metrics.items()
        }
        
        # Compute task-specific metrics if applicable
        if compute_task_metrics and all_predictions:
            from src.evaluation.metrics import compute_metrics
            # Get tokenizer for token-level label decoding
            tokenizer = self.eval_dataloader.dataset.tokenizer if hasattr(self.eval_dataloader.dataset, 'tokenizer') else None
            task_metrics = compute_metrics(
                predictions=all_predictions,
                references=all_references,
                task_config=task_config,
                tokenizer=tokenizer,
            )
            # Keep token-level accuracy as primary; store generation-based under separate key
            eval_metrics["generation_accuracy"] = task_metrics.get("accuracy")
            for k, v in task_metrics.items():
                if k != "accuracy":
                    eval_metrics[k] = v
            logger.info(f"Task-specific metrics: {task_metrics}")
        
        return eval_metrics
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = (self.checkpoint_dir / name).resolve()
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = checkpoint_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Save model state (only if trainable)
        if any(p.requires_grad for p in self.model.parameters()):
            model_path = checkpoint_path / "model.pt"
            torch.save(self.model.state_dict(), model_path)
        
        # Save LoRA module (to final_model/ and to run root for downstream scripts)
        if self.lora_module:
            lora_path = checkpoint_path / "lora_module.pt"
            torch.save(self.lora_module.state_dict(), lora_path)
            if not lora_path.exists() or lora_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Failed to save lora_module.pt at {lora_path} (missing or empty). "
                    "Check disk space and permissions."
                )
            # Also save to run root so composer/evaluate find it when given the run dir
            run_root_lora = self.checkpoint_dir / "lora_module.pt"
            if run_root_lora.resolve() != lora_path.resolve():
                torch.save(self.lora_module.state_dict(), run_root_lora)
                logger.info(f"Saved lora_module.pt to run root {run_root_lora}")
        
        # Save router
        if self.router:
            router_path = checkpoint_path / "router.pt"
            torch.save(self.router.state_dict(), router_path)
        
        # Save optimizer and scheduler
        optimizer_path = checkpoint_path / "optimizer.pt"
        checkpoint_data = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step,
            "best_metric": self.best_metric,
        }
        
        # Include latest evaluation metrics if available
        if self.evaluation_history:
            checkpoint_data["latest_eval_metrics"] = self.evaluation_history[-1]["metrics"]
            checkpoint_data["latest_eval_step"] = self.evaluation_history[-1]["step"]
        
        torch.save(checkpoint_data, optimizer_path)
        
        # Save checkpoint metadata as JSON for easy inspection
        metadata = {
            "checkpoint_name": name,
            "step": self.global_step,
            "best_metric": self.best_metric,
            "has_model": any(p.requires_grad for p in self.model.parameters()),
            "has_lora": self.lora_module is not None,
            "has_router": self.router is not None,
        }
        if self.evaluation_history:
            metadata["latest_eval_metrics"] = self.evaluation_history[-1]["metrics"]
            metadata["latest_eval_step"] = self.evaluation_history[-1]["step"]
        
        metadata_path = checkpoint_path / "checkpoint_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load model
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load LoRA
        lora_path = checkpoint_path / "lora_module.pt"
        if lora_path.exists() and self.lora_module:
            self.lora_module.load_state_dict(torch.load(lora_path, map_location=self.device))
        
        # Load router
        router_path = checkpoint_path / "router.pt"
        if router_path.exists() and self.router:
            self.router.load_state_dict(torch.load(router_path, map_location=self.device))
        
        # Load optimizer
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            checkpoint = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.global_step = checkpoint["step"]
            self.best_metric = checkpoint.get("best_metric", float("-inf"))
        
        logger.info(f"Loaded checkpoint from step {self.global_step}")
