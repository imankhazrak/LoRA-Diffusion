"""Base Masked Diffusion Language Model.

Implements a discrete diffusion model for text with masking corruption.
"""

import math
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutput

if TYPE_CHECKING:
    from .lora_modules import MultiTaskLoRAComposer


class NoiseSchedule:
    """Manages the noise schedule for diffusion process."""
    
    def __init__(
        self,
        num_steps: int,
        schedule_type: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Args:
            num_steps: Total diffusion steps T
            schedule_type: "linear" or "cosine"
            beta_start: Starting beta value
            beta_end: Ending beta value
        """
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in improved DDPM paper."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


class MaskedDiffusionTransformer(nn.Module):
    """
    Masked Diffusion Language Model.
    
    Uses a transformer to denoise masked tokens at each diffusion step.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Extract config
        self.hidden_dim = config["model"]["hidden_dim"]
        self.num_layers = config["model"]["num_layers"]
        self.num_heads = config["model"]["num_heads"]
        self.vocab_size = config["model"]["vocab_size"]
        self.max_seq_length = config["model"]["max_seq_length"]
        
        # Diffusion config
        self.num_diffusion_steps = config["diffusion"]["num_steps"]
        self.mask_token_id = config["diffusion"]["mask_token_id"]
        
        # Initialize noise schedule
        self.noise_schedule = NoiseSchedule(
            num_steps=self.num_diffusion_steps,
            schedule_type=config["diffusion"]["schedule"],
            beta_start=config["diffusion"]["beta_start"],
            beta_end=config["diffusion"]["beta_end"],
        )
        
        # Build transformer backbone using BERT architecture
        bert_config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=config["model"]["ffn_dim"],
            max_position_embeddings=self.max_seq_length,
            hidden_dropout_prob=config["model"]["dropout"],
            attention_probs_dropout_prob=config["model"]["attention_dropout"],
        )
        self.transformer = BertModel(bert_config)
        
        # Time step embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        
        # Output head to predict x0 logits
        self.output_head = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal time embeddings.
        
        Args:
            timesteps: (batch_size,) tensor of timesteps
            
        Returns:
            (batch_size, hidden_dim) time embeddings
        """
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.hidden_dim % 2 == 1:  # Pad if odd dimension
            emb = F.pad(emb, (0, 1))
        
        return self.time_embed(emb)
    
    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion process (add noise).
        
        Args:
            x0: (batch_size, seq_len) clean token IDs
            t: (batch_size,) timesteps
            
        Returns:
            xt: (batch_size, seq_len) corrupted token IDs
            mask: (batch_size, seq_len) boolean mask of corrupted positions
        """
        batch_size, seq_len = x0.shape
        device = x0.device
        
        # Get corruption probability for each timestep
        alphas_cumprod = self.noise_schedule.alphas_cumprod.to(device)
        beta_t = 1.0 - alphas_cumprod[t]  # (batch_size,)
        
        # Sample which tokens to mask
        # Probability of masking each token is beta_t
        mask_prob = beta_t[:, None].expand(-1, seq_len)  # (batch_size, seq_len)
        mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
        
        # Create corrupted input
        xt = x0.clone()
        xt[mask] = self.mask_token_id
        
        return xt, mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        instruction_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict x0 distribution from xt.
        
        Args:
            input_ids: (batch_size, seq_len) noisy token IDs (xt)
            timesteps: (batch_size,) diffusion timesteps
            attention_mask: (batch_size, seq_len) attention mask
            instruction_embedding: (batch_size, hidden_dim) optional instruction conditioning
            
        Returns:
            (batch_size, seq_len, vocab_size) logits for x0 prediction
        """
        batch_size, seq_len = input_ids.shape
        
        # Get time embeddings
        time_emb = self.get_time_embedding(timesteps)  # (batch_size, hidden_dim)
        
        # Get transformer hidden states
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # Add time embedding to each position
        time_emb_expanded = time_emb[:, None, :].expand(-1, seq_len, -1)
        hidden_states = hidden_states + time_emb_expanded
        
        # Optionally add instruction conditioning
        if instruction_embedding is not None:
            inst_emb_expanded = instruction_embedding[:, None, :].expand(-1, seq_len, -1)
            hidden_states = hidden_states + inst_emb_expanded
        
        # Predict x0 logits
        logits = self.output_head(hidden_states)
        
        return logits
    
    def compute_loss(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        instruction_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute denoising loss.
        
        Args:
            x0: (batch_size, seq_len) clean token IDs
            timesteps: (batch_size,) diffusion timesteps
            attention_mask: (batch_size, seq_len) attention mask
            instruction_embedding: (batch_size, hidden_dim) optional instruction conditioning
            
        Returns:
            loss: scalar loss
            metrics: dictionary of metrics
        """
        # Apply forward diffusion
        xt, mask = self.forward_diffusion(x0, timesteps)
        
        # Predict x0 from xt
        logits = self.forward(
            input_ids=xt,
            timesteps=timesteps,
            attention_mask=attention_mask,
            instruction_embedding=instruction_embedding,
        )
        
        # Compute cross-entropy loss only on masked positions
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            x0.reshape(-1),
            reduction="none",
        )
        loss = loss.reshape(x0.shape)
        
        # Apply mask: only compute loss on corrupted tokens
        if attention_mask is not None:
            mask = mask & attention_mask.bool()
        
        loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        
        # Compute accuracy
        with torch.no_grad():
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == x0) & mask
            accuracy = correct.float().sum() / mask.float().sum()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        instruction_embedding: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Sample from the model using reverse diffusion.
        
        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            instruction_embedding: (batch_size, hidden_dim) optional instruction conditioning
            device: Device to run on
            
        Returns:
            (batch_size, seq_len) generated token IDs
        """
        # Start from pure noise (all masked)
        xt = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # Reverse diffusion
        for t in reversed(range(self.num_diffusion_steps)):
            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            # Predict x0
            logits = self.forward(
                input_ids=xt,
                timesteps=timesteps,
                instruction_embedding=instruction_embedding,
            )
            
            # Sample x0
            x0_pred = torch.multinomial(
                F.softmax(logits.reshape(-1, self.vocab_size), dim=-1),
                num_samples=1,
            ).reshape(batch_size, seq_len)
            
            if t > 0:
                # Compute q(x_{t-1} | x_t, x0)
                # For masked diffusion, gradually unmask tokens
                alphas_cumprod = self.noise_schedule.alphas_cumprod.to(device)
                beta_t = 1.0 - alphas_cumprod[t]
                
                # Probability of keeping token masked at t-1
                keep_mask_prob = beta_t * (1.0 - alphas_cumprod[t - 1]) / (1.0 - alphas_cumprod[t])
                keep_mask = torch.rand(batch_size, seq_len, device=device) < keep_mask_prob
                
                # Update xt -> x_{t-1}
                xt = torch.where(keep_mask, self.mask_token_id, x0_pred)
            else:
                xt = x0_pred
        
        return xt
    
    @torch.no_grad()
    def sample_with_composition(
        self,
        batch_size: int,
        seq_len: int,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        composer: "MultiTaskLoRAComposer",
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Sample from the model using multi-task LoRA composition.
        
        At each diffusion step t:
        1. Get base hidden states: h_t = transformer(x_t, t)
        2. Get composed perturbation: δ = composer(h_t, t, instruction_ids, ...)
        3. Apply: h_t' = h_t + δ
        4. Get logits: logits = output_head(h_t')
        
        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            instruction_ids: (batch_size, inst_len) instruction token IDs
            instruction_mask: (batch_size, inst_len) instruction attention mask
            composer: MultiTaskLoRAComposer instance
            device: Device to run on
            
        Returns:
            (batch_size, seq_len) generated token IDs
        """
        # Start from pure noise (all masked)
        xt = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # Move instruction tensors to device
        instruction_ids = instruction_ids.to(device)
        instruction_mask = instruction_mask.to(device)
        
        # Reverse diffusion
        for t in reversed(range(self.num_diffusion_steps)):
            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            # Get base hidden representations
            hidden_states = self.get_representation(
                input_ids=xt,
                timesteps=timesteps,
                attention_mask=None,  # All tokens are valid at this point
            )
            
            # Get composed LoRA perturbation
            delta = composer(
                hidden_states=hidden_states,
                timesteps=timesteps,
                instruction_ids=instruction_ids,
                instruction_mask=instruction_mask,
            )
            
            # Apply perturbation: h_t' = h_t + δ
            perturbed_hidden = hidden_states + delta
            
            # Get logits from perturbed hidden states
            logits = self.output_head(perturbed_hidden)
            
            # Sample x0
            x0_pred = torch.multinomial(
                F.softmax(logits.reshape(-1, self.vocab_size), dim=-1),
                num_samples=1,
            ).reshape(batch_size, seq_len)
            
            if t > 0:
                # Compute q(x_{t-1} | x_t, x0)
                # For masked diffusion, gradually unmask tokens
                alphas_cumprod = self.noise_schedule.alphas_cumprod.to(device)
                beta_t = 1.0 - alphas_cumprod[t]
                
                # Probability of keeping token masked at t-1
                keep_mask_prob = beta_t * (1.0 - alphas_cumprod[t - 1]) / (1.0 - alphas_cumprod[t])
                keep_mask = torch.rand(batch_size, seq_len, device=device) < keep_mask_prob
                
                # Update xt -> x_{t-1}
                xt = torch.where(keep_mask, self.mask_token_id, x0_pred)
            else:
                xt = x0_pred
        
        return xt
    
    def get_representation(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get hidden representations for LoRA-Diffusion.
        
        Args:
            input_ids: (batch_size, seq_len) noisy token IDs
            timesteps: (batch_size,) diffusion timesteps
            attention_mask: (batch_size, seq_len) attention mask
            
        Returns:
            (batch_size, seq_len, hidden_dim) hidden states
        """
        # Get time embeddings
        time_emb = self.get_time_embedding(timesteps)
        
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs.last_hidden_state
        
        # Add time embedding
        time_emb_expanded = time_emb[:, None, :].expand(-1, hidden_states.size(1), -1)
        hidden_states = hidden_states + time_emb_expanded
        
        return hidden_states
