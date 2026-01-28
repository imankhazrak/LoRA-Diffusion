"""Unit tests for base diffusion model."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_diffusion import (
    NoiseSchedule,
    MaskedDiffusionTransformer,
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
            "gradient_checkpointing": False,
            "compile": False,
        },
        "diffusion": {
            "num_steps": 100,
            "schedule": "cosine",
            "forward_type": "mask",
            "mask_token_id": 103,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        },
    }


class TestNoiseSchedule:
    """Test NoiseSchedule class."""
    
    def test_linear_schedule(self):
        """Test linear noise schedule."""
        num_steps = 100
        schedule = NoiseSchedule(
            num_steps=num_steps,
            schedule_type="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )
        
        assert schedule.betas.shape == (num_steps,)
        assert schedule.betas[0] < schedule.betas[-1]  # Increasing
        assert torch.all(schedule.betas >= 0.0)
        assert torch.all(schedule.betas <= 1.0)
    
    def test_cosine_schedule(self):
        """Test cosine noise schedule."""
        num_steps = 100
        schedule = NoiseSchedule(
            num_steps=num_steps,
            schedule_type="cosine",
        )
        
        assert schedule.betas.shape == (num_steps,)
        assert torch.all(schedule.betas >= 0.0)
        assert torch.all(schedule.betas <= 1.0)
    
    def test_alphas_computation(self):
        """Test alpha values are computed correctly."""
        schedule = NoiseSchedule(num_steps=100, schedule_type="linear")
        
        # alphas = 1 - betas
        expected_alphas = 1.0 - schedule.betas
        assert torch.allclose(schedule.alphas, expected_alphas)
        
        # alphas_cumprod should be cumulative product
        expected_cumprod = torch.cumprod(schedule.alphas, dim=0)
        assert torch.allclose(schedule.alphas_cumprod, expected_cumprod)


class TestMaskedDiffusionTransformer:
    """Test MaskedDiffusionTransformer."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        assert model.hidden_dim == config["model"]["hidden_dim"]
        assert model.num_layers == config["model"]["num_layers"]
        assert model.vocab_size == config["model"]["vocab_size"]
        assert model.num_diffusion_steps == config["diffusion"]["num_steps"]
    
    def test_time_embedding(self):
        """Test time embedding generation."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        batch_size = 4
        timesteps = torch.randint(0, 100, (batch_size,))
        
        time_emb = model.get_time_embedding(timesteps)
        
        assert time_emb.shape == (batch_size, config["model"]["hidden_dim"])
        assert not torch.isnan(time_emb).any()
        assert not torch.isinf(time_emb).any()
    
    def test_forward_diffusion(self):
        """Test forward diffusion process (adding noise)."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        batch_size = 4
        seq_len = 16
        vocab_size = config["model"]["vocab_size"]
        
        # Create clean input
        x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, 100, (batch_size,))
        
        # Apply forward diffusion
        xt, mask = model.forward_diffusion(x0, timesteps)
        
        # Check output shapes
        assert xt.shape == x0.shape
        assert mask.shape == x0.shape
        assert mask.dtype == torch.bool
        
        # Check that masked positions have mask token
        assert torch.all(xt[mask] == config["diffusion"]["mask_token_id"])
        
        # Check that unmasked positions are unchanged
        assert torch.all(xt[~mask] == x0[~mask])
    
    def test_forward_pass(self):
        """Test forward pass (denoising)."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        batch_size = 4
        seq_len = 16
        vocab_size = config["model"]["vocab_size"]
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, 100, (batch_size,))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        logits = model.forward(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        
        # Check output shape
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_forward_with_instruction(self):
        """Test forward pass with instruction conditioning."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        batch_size = 4
        seq_len = 16
        vocab_size = config["model"]["vocab_size"]
        hidden_dim = config["model"]["hidden_dim"]
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, 100, (batch_size,))
        instruction_embedding = torch.randn(batch_size, hidden_dim)
        
        # Forward pass with instruction
        logits = model.forward(
            input_ids=input_ids,
            timesteps=timesteps,
            instruction_embedding=instruction_embedding,
        )
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(logits).any()
    
    def test_compute_loss(self):
        """Test loss computation."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        batch_size = 4
        seq_len = 16
        vocab_size = config["model"]["vocab_size"]
        
        x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, 100, (batch_size,))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Compute loss
        loss, metrics = model.compute_loss(
            x0=x0,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        
        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
        # Check metrics
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_get_representation(self):
        """Test getting hidden representations."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        batch_size = 4
        seq_len = 16
        vocab_size = config["model"]["vocab_size"]
        hidden_dim = config["model"]["hidden_dim"]
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, 100, (batch_size,))
        
        # Get representations
        hidden_states = model.get_representation(
            input_ids=input_ids,
            timesteps=timesteps,
        )
        
        assert hidden_states.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(hidden_states).any()
    
    def test_sample_generation(self):
        """Test sampling from the model."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        model.eval()
        
        batch_size = 2
        seq_len = 8
        vocab_size = config["model"]["vocab_size"]
        
        # Generate samples
        with torch.no_grad():
            samples = model.sample(
                batch_size=batch_size,
                seq_len=seq_len,
                device="cpu",
            )
        
        # Check output
        assert samples.shape == (batch_size, seq_len)
        assert samples.dtype == torch.long
        assert torch.all(samples >= 0)
        assert torch.all(samples < vocab_size)
        
        # Should not be all mask tokens
        mask_token_id = config["diffusion"]["mask_token_id"]
        assert not torch.all(samples == mask_token_id)
    
    def test_different_timesteps_in_batch(self):
        """Test handling batches with different timesteps."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        batch_size = 4
        seq_len = 16
        vocab_size = config["model"]["vocab_size"]
        
        # Different timesteps for each sample
        x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
        timesteps = torch.tensor([10, 30, 60, 90])  # Different phases
        
        # Should work without errors
        xt, mask = model.forward_diffusion(x0, timesteps)
        assert xt.shape == x0.shape
        
        logits = model.forward(input_ids=xt, timesteps=timesteps)
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        batch_size = 2
        seq_len = 8
        vocab_size = config["model"]["vocab_size"]
        
        x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, 100, (batch_size,))
        
        # Compute loss
        loss, _ = model.compute_loss(x0=x0, timesteps=timesteps)
        
        # Backward
        loss.backward()
        
        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                assert not torch.isnan(param.grad).any()
                break
        
        assert has_gradients, "No gradients computed"


class TestDiffusionProperties:
    """Test mathematical properties of diffusion process."""
    
    def test_noise_increases_with_timestep(self):
        """Test that noise increases as t increases."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        
        batch_size = 100
        seq_len = 32
        vocab_size = config["model"]["vocab_size"]
        
        x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test different timesteps
        mask_ratios = []
        for t in [10, 30, 50, 70, 90]:
            timesteps = torch.full((batch_size,), t)
            _, mask = model.forward_diffusion(x0, timesteps)
            mask_ratio = mask.float().mean().item()
            mask_ratios.append(mask_ratio)
        
        # Mask ratio should generally increase with t
        # (allowing some noise due to stochastic sampling)
        assert mask_ratios[-1] > mask_ratios[0], \
            f"Mask ratio should increase: {mask_ratios}"
    
    def test_forward_reverse_consistency(self):
        """Test that forward diffusion produces valid inputs for reverse."""
        config = get_test_config()
        model = MaskedDiffusionTransformer(config)
        model.eval()
        
        batch_size = 4
        seq_len = 16
        vocab_size = config["model"]["vocab_size"]
        
        x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(20, 80, (batch_size,))  # Mid-range
        
        # Forward diffusion
        xt, mask = model.forward_diffusion(x0, timesteps)
        
        # Should be able to run reverse step
        with torch.no_grad():
            logits = model.forward(input_ids=xt, timesteps=timesteps)
            predictions = logits.argmax(dim=-1)
        
        # Predictions should be valid token IDs
        assert torch.all(predictions >= 0)
        assert torch.all(predictions < vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
