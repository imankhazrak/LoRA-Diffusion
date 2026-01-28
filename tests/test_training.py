"""Unit tests for training pipeline."""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_diffusion import MaskedDiffusionTransformer
from src.models.lora_modules import LoRADiffusionModule
from src.training.losses import compute_diffusion_loss
from src.training.trainer import DiffusionTrainer
from torch.utils.data import DataLoader, TensorDataset


def get_test_config():
    """Get minimal test configuration."""
    return {
        "model": {
            "hidden_dim": 128,
            "vocab_size": 500,
            "num_layers": 4,
            "num_heads": 4,
            "ffn_dim": 512,
            "max_seq_length": 64,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "gradient_checkpointing": False,
            "compile": False,
        },
        "diffusion": {
            "num_steps": 50,
            "schedule": "cosine",
            "forward_type": "mask",
            "mask_token_id": 103,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        },
        "lora": {
            "rank_early": 32,
            "rank_mid": 16,
            "rank_late": 8,
            "scaling_early": 1.0,
            "scaling_mid": 0.5,
            "scaling_late": 0.25,
            "num_modules": 2,
            "rank_reg_weight": 0.01,
            "orth_reg_weight": 0.001,
            "instruction_encoder_hidden": 64,
            "instruction_encoder_layers": 2,
        },
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "lr_scheduler": "cosine",
            "warmup_steps": 10,
            "max_steps": 100,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "eval_batch_size": 8,
            "mixed_precision": "no",
            "logging_steps": 10,
            "eval_frequency": 50,
            "save_frequency": 50,
            "save_total_limit": 2,
            "seed": 42,
            "deterministic": True,
        },
        "output": {
            "base_dir": "./test_outputs",
            "checkpoint_dir": "./test_checkpoints",
            "log_dir": "./test_logs",
            "save_predictions": True,
            "save_metrics": True,
        },
        "data": {
            "cache_dir": "./test_cache",
            "num_workers": 0,
            "pin_memory": False,
        },
        "metrics": {
            "primary": "accuracy",
            "secondary": ["loss"],
        },
        "method": {
            "activation": "relu",
            "init_scale": 0.01,
        },
    }


def create_dummy_batch(batch_size=4, seq_len=16, vocab_size=500):
    """Create a dummy batch for testing."""
    return {
        "instruction_ids": torch.randint(0, vocab_size, (batch_size, 10)),
        "instruction_mask": torch.ones(batch_size, 10),
        "target_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "target_mask": torch.ones(batch_size, seq_len),
    }


def create_dummy_dataloader(num_samples=20, batch_size=4):
    """Create a dummy dataloader for testing."""
    # Create dummy tensors
    instruction_ids = torch.randint(0, 500, (num_samples, 10))
    instruction_mask = torch.ones(num_samples, 10)
    target_ids = torch.randint(0, 500, (num_samples, 16))
    target_mask = torch.ones(num_samples, 16)
    
    # Create TensorDataset
    dataset = TensorDataset(
        instruction_ids,
        instruction_mask,
        target_ids,
        target_mask,
    )
    
    def collate_fn(batch):
        return {
            "instruction_ids": torch.stack([x[0] for x in batch]),
            "instruction_mask": torch.stack([x[1] for x in batch]),
            "target_ids": torch.stack([x[2] for x in batch]),
            "target_mask": torch.stack([x[3] for x in batch]),
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )


class TestDiffusionLoss:
    """Test loss computation."""
    
    def test_loss_without_lora(self):
        """Test loss computation without LoRA module."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        batch = create_dummy_batch()
        
        loss, metrics = compute_diffusion_loss(
            model=model,
            batch=batch,
            lora_module=None,
            config=config,
        )
        
        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
        # Check metrics
        assert "loss" in metrics
        assert "ce_loss" in metrics
        assert "accuracy" in metrics
        assert metrics["reg_loss"] == 0.0  # No LoRA
    
    def test_loss_with_lora(self):
        """Test loss computation with LoRA module."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        lora_module = LoRADiffusionModule(config)
        
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
        
        batch = create_dummy_batch()
        
        loss, metrics = compute_diffusion_loss(
            model=model,
            batch=batch,
            lora_module=lora_module,
            config=config,
        )
        
        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        
        # Check metrics
        assert "loss" in metrics
        assert "ce_loss" in metrics
        assert "reg_loss" in metrics
        assert metrics["reg_loss"] >= 0  # Should have regularization
        assert "accuracy" in metrics
    
    def test_loss_gradient_flow(self):
        """Test that gradients flow through loss."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        lora_module = LoRADiffusionModule(config)
        
        batch = create_dummy_batch()
        
        loss, _ = compute_diffusion_loss(
            model=model,
            batch=batch,
            lora_module=lora_module,
            config=config,
        )
        
        # Backward
        loss.backward()
        
        # Check LoRA gradients exist
        has_gradients = False
        for param in lora_module.parameters():
            if param.grad is not None:
                has_gradients = True
                assert not torch.isnan(param.grad).any()
        
        assert has_gradients


class TestTrainer:
    """Test DiffusionTrainer."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = get_test_config()
        self.config["output"]["base_dir"] = str(Path(self.temp_dir) / "outputs")
        self.config["output"]["checkpoint_dir"] = str(Path(self.temp_dir) / "checkpoints")
        self.config["output"]["log_dir"] = str(Path(self.temp_dir) / "logs")
    
    def teardown_method(self):
        """Cleanup after each test."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = MaskedDiffusionTransformer(self.config)
        lora_module = LoRADiffusionModule(self.config)
        
        train_loader = create_dummy_dataloader(num_samples=20, batch_size=4)
        eval_loader = create_dummy_dataloader(num_samples=10, batch_size=4)
        
        optimizer = torch.optim.AdamW(lora_module.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        trainer = DiffusionTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self.config,
            lora_module=lora_module,
            device="cpu",
        )
        
        assert trainer.global_step == 0
        assert trainer.max_steps == self.config["training"]["max_steps"]
        assert trainer.device == "cpu"
    
    def test_trainer_short_training(self):
        """Test running a few training steps."""
        # Very short training run
        self.config["training"]["max_steps"] = 5
        self.config["training"]["logging_steps"] = 2
        self.config["training"]["eval_frequency"] = 10  # No eval
        self.config["training"]["save_frequency"] = 10  # No save
        
        model = MaskedDiffusionTransformer(self.config)
        lora_module = LoRADiffusionModule(self.config)
        
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
        
        train_loader = create_dummy_dataloader(num_samples=20, batch_size=4)
        eval_loader = create_dummy_dataloader(num_samples=10, batch_size=4)
        
        optimizer = torch.optim.AdamW(lora_module.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        
        trainer = DiffusionTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self.config,
            lora_module=lora_module,
            device="cpu",
        )
        
        # Train
        trainer.train()
        
        # Check training completed
        assert trainer.global_step == 5
        assert len(trainer.training_history) > 0
    
    def test_evaluation(self):
        """Test evaluation."""
        model = MaskedDiffusionTransformer(self.config)
        lora_module = LoRADiffusionModule(self.config)
        
        train_loader = create_dummy_dataloader(num_samples=20, batch_size=4)
        eval_loader = create_dummy_dataloader(num_samples=10, batch_size=4)
        
        optimizer = torch.optim.AdamW(lora_module.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        trainer = DiffusionTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self.config,
            lora_module=lora_module,
            device="cpu",
        )
        
        # Run evaluation
        metrics = trainer.evaluate()
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        model = MaskedDiffusionTransformer(self.config)
        lora_module = LoRADiffusionModule(self.config)
        
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
        
        train_loader = create_dummy_dataloader(num_samples=20, batch_size=4)
        eval_loader = create_dummy_dataloader(num_samples=10, batch_size=4)
        
        optimizer = torch.optim.AdamW(lora_module.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        trainer = DiffusionTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self.config,
            lora_module=lora_module,
            device="cpu",
        )
        
        # Save checkpoint
        trainer.global_step = 42
        trainer.best_metric = 0.85
        checkpoint_name = "test_checkpoint"
        trainer.save_checkpoint(checkpoint_name)
        
        # Check files exist
        checkpoint_path = Path(self.config["output"]["checkpoint_dir"]) / checkpoint_name
        assert (checkpoint_path / "config.json").exists()
        assert (checkpoint_path / "lora_module.pt").exists()
        assert (checkpoint_path / "optimizer.pt").exists()
        
        # Create new trainer and load
        new_lora = LoRADiffusionModule(self.config)
        new_optimizer = torch.optim.AdamW(new_lora.parameters(), lr=1e-4)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=100)
        
        new_trainer = DiffusionTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            config=self.config,
            lora_module=new_lora,
            device="cpu",
        )
        
        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Check state restored
        assert new_trainer.global_step == 42
        assert new_trainer.best_metric == 0.85
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        self.config["training"]["gradient_accumulation_steps"] = 2
        self.config["training"]["max_steps"] = 4
        self.config["training"]["logging_steps"] = 100  # No logging
        
        model = MaskedDiffusionTransformer(self.config)
        lora_module = LoRADiffusionModule(self.config)
        
        for param in model.parameters():
            param.requires_grad = False
        
        train_loader = create_dummy_dataloader(num_samples=20, batch_size=2)
        eval_loader = create_dummy_dataloader(num_samples=10, batch_size=2)
        
        optimizer = torch.optim.AdamW(lora_module.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        
        trainer = DiffusionTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self.config,
            lora_module=lora_module,
            device="cpu",
        )
        
        # Should complete without errors
        trainer.train()
        assert trainer.global_step == 4


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_full_pipeline_without_lora(self):
        """Test full training pipeline without LoRA (full fine-tuning)."""
        config = get_test_config()
        config["training"]["max_steps"] = 3
        config["training"]["logging_steps"] = 100
        config["training"]["eval_frequency"] = 100
        
        model = MaskedDiffusionTransformer(config)
        
        train_loader = create_dummy_dataloader(num_samples=12, batch_size=4)
        eval_loader = create_dummy_dataloader(num_samples=8, batch_size=4)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        
        temp_dir = tempfile.mkdtemp()
        config["output"]["base_dir"] = str(Path(temp_dir) / "outputs")
        config["output"]["checkpoint_dir"] = str(Path(temp_dir) / "checkpoints")
        
        try:
            trainer = DiffusionTrainer(
                model=model,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                lora_module=None,  # Full fine-tuning
                device="cpu",
            )
            
            trainer.train()
            
            assert trainer.global_step == 3
        finally:
            shutil.rmtree(temp_dir)
    
    def test_full_pipeline_with_lora(self):
        """Test full training pipeline with LoRA."""
        config = get_test_config()
        config["training"]["max_steps"] = 3
        config["training"]["logging_steps"] = 100
        config["training"]["eval_frequency"] = 100
        
        model = MaskedDiffusionTransformer(config)
        lora_module = LoRADiffusionModule(config)
        
        # Freeze base
        for param in model.parameters():
            param.requires_grad = False
        
        train_loader = create_dummy_dataloader(num_samples=12, batch_size=4)
        eval_loader = create_dummy_dataloader(num_samples=8, batch_size=4)
        
        optimizer = torch.optim.AdamW(lora_module.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        
        temp_dir = tempfile.mkdtemp()
        config["output"]["base_dir"] = str(Path(temp_dir) / "outputs")
        config["output"]["checkpoint_dir"] = str(Path(temp_dir) / "checkpoints")
        
        try:
            trainer = DiffusionTrainer(
                model=model,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                lora_module=lora_module,
                device="cpu",
            )
            
            trainer.train()
            
            assert trainer.global_step == 3
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
