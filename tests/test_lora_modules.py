"""Unit tests for LoRA-Diffusion modules."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lora_modules import (
    TrajectoryLoRAAdapter,
    LoRADiffusionModule,
    InstructionEncoder,
)


def get_test_config():
    """Get minimal test configuration."""
    return {
        "model": {
            "hidden_dim": 256,
            "vocab_size": 1000,
            "num_layers": 6,
            "num_heads": 8,
            "ffn_dim": 1024,
            "max_seq_length": 128,
            "dropout": 0.1,
            "attention_dropout": 0.1,
        },
        "diffusion": {
            "num_steps": 100,
            "schedule": "cosine",
            "forward_type": "mask",
            "mask_token_id": 103,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        },
        "lora": {
            "rank_early": 64,
            "rank_mid": 32,
            "rank_late": 8,
            "scaling_early": 1.0,
            "scaling_mid": 0.5,
            "scaling_late": 0.25,
            "num_modules": 2,
            "rank_reg_weight": 0.01,
            "orth_reg_weight": 0.001,
            "instruction_encoder_hidden": 128,
            "instruction_encoder_layers": 2,
        },
        "method": {
            "activation": "relu",
            "init_scale": 0.01,
        },
    }


class TestInstructionEncoder:
    """Test InstructionEncoder."""
    
    def test_forward(self):
        """Test forward pass."""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        hidden_dim = 256
        output_dim = 128
        
        encoder = InstructionEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = encoder(input_ids, attention_mask)
        
        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()


class TestTrajectoryLoRAAdapter:
    """Test TrajectoryLoRAAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        hidden_dim = 256
        rank = 32
        instruction_dim = 128
        
        adapter = TrajectoryLoRAAdapter(
            hidden_dim=hidden_dim,
            rank=rank,
            instruction_dim=instruction_dim,
        )
        
        # Check shapes
        assert adapter.down_proj.weight.shape == (rank, hidden_dim)
        assert adapter.up_proj_base.weight.shape == (hidden_dim, rank)
        assert adapter.up_proj_cond_scale.weight.shape == (rank, instruction_dim)
        assert adapter.up_proj_cond_shift.weight.shape == (hidden_dim, instruction_dim)
    
    def test_forward(self):
        """Test forward pass."""
        batch_size = 4
        seq_len = 16
        hidden_dim = 256
        rank = 32
        instruction_dim = 128
        
        adapter = TrajectoryLoRAAdapter(
            hidden_dim=hidden_dim,
            rank=rank,
            instruction_dim=instruction_dim,
        )
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        instruction_emb = torch.randn(batch_size, instruction_dim)
        
        output = adapter(hidden_states, instruction_emb)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(output).any()


class TestLoRADiffusionModule:
    """Test LoRADiffusionModule."""
    
    def test_initialization(self):
        """Test module initialization."""
        config = get_test_config()
        module = LoRADiffusionModule(config)
        
        # Check adapters created
        assert len(module.adapters_early) == config["lora"]["num_modules"]
        assert len(module.adapters_mid) == config["lora"]["num_modules"]
        assert len(module.adapters_late) == config["lora"]["num_modules"]
        
        # Check ranks
        assert module.adapters_early[0].rank == config["lora"]["rank_early"]
        assert module.adapters_mid[0].rank == config["lora"]["rank_mid"]
        assert module.adapters_late[0].rank == config["lora"]["rank_late"]
    
    def test_step_config(self):
        """Test step-adaptive configuration."""
        config = get_test_config()
        module = LoRADiffusionModule(config)
        
        # Test early phase
        adapters, scaling = module.get_step_config(timestep=80)
        assert adapters == module.adapters_early
        assert scaling == config["lora"]["scaling_early"]
        
        # Test middle phase
        adapters, scaling = module.get_step_config(timestep=50)
        assert adapters == module.adapters_mid
        assert scaling == config["lora"]["scaling_mid"]
        
        # Test late phase
        adapters, scaling = module.get_step_config(timestep=20)
        assert adapters == module.adapters_late
        assert scaling == config["lora"]["scaling_late"]
    
    def test_forward(self):
        """Test forward pass."""
        config = get_test_config()
        module = LoRADiffusionModule(config)
        
        batch_size = 4
        seq_len = 16
        hidden_dim = config["model"]["hidden_dim"]
        vocab_size = config["model"]["vocab_size"]
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        timesteps = torch.randint(0, 100, (batch_size,))
        instruction_ids = torch.randint(0, vocab_size, (batch_size, 10))
        instruction_mask = torch.ones(batch_size, 10)
        
        delta = module(
            hidden_states=hidden_states,
            timesteps=timesteps,
            instruction_ids=instruction_ids,
            instruction_mask=instruction_mask,
        )
        
        assert delta.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(delta).any()
    
    def test_regularization(self):
        """Test regularization loss."""
        config = get_test_config()
        module = LoRADiffusionModule(config)
        
        reg_loss = module.compute_regularization_loss()
        
        assert isinstance(reg_loss, (int, float, torch.Tensor))
        if isinstance(reg_loss, torch.Tensor):
            assert reg_loss.item() >= 0
            assert not torch.isnan(reg_loss)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        config = get_test_config()
        module = LoRADiffusionModule(config)
        
        counts = module.count_parameters()
        
        assert "instruction_encoder" in counts
        assert "adapters_early" in counts
        assert "adapters_mid" in counts
        assert "adapters_late" in counts
        assert "total" in counts
        assert counts["total"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
