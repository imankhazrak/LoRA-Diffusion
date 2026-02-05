"""LoRA-Diffusion: Trajectory-level Low-Rank Adaptation Modules.

Implements low-rank perturbations to the denoising trajectory.
"""

import math
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionEncoder(nn.Module):
    """
    Lightweight encoder for instruction conditioning.
    
    Encodes instruction text into a hidden representation used to condition
    the up-projection matrix A_i(c).
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project to output dimension
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode instruction text.
        
        Args:
            input_ids: (batch_size, seq_len) instruction token IDs
            attention_mask: (batch_size, seq_len) attention mask
            
        Returns:
            (batch_size, output_dim) instruction embedding
        """
        # Embed tokens
        embeds = self.embedding(input_ids)  # (batch_size, seq_len, hidden_dim)
        
        # Apply transformer
        if attention_mask is not None:
            # Convert to transformer mask format (inverted)
            mask = attention_mask == 0
        else:
            mask = None
        
        hidden_states = self.transformer(embeds, src_key_padding_mask=mask)
        
        # Mean pool over sequence length
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Project to output
        output = self.output_proj(pooled)
        
        return output


class TaskRouter(nn.Module):
    """
    Lightweight router for multi-task composition.
    
    Maps instruction embeddings to task weights: w = softmax(Router(Enc(c)))
    Architecture: 2-layer MLP with 512 hidden units (default)
    
    From paper Section 3.5:
    - Input: Instruction embedding Enc(c) ∈ R^d
    - Hidden: 2-layer MLP with 512 hidden units
    - Output: M-dimensional logits → softmax probabilities
    - Parameters: ~1M (negligible compared to base model)
    """
    
    def __init__(
        self,
        instruction_dim: int,
        num_tasks: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            instruction_dim: Dimension of instruction embedding (from InstructionEncoder)
            num_tasks: Number of tasks M
            hidden_dim: Hidden dimension of MLP (default: 512)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.instruction_dim = instruction_dim
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        
        # 2-layer MLP: instruction_dim → hidden_dim → num_tasks
        self.mlp = nn.Sequential(
            nn.Linear(instruction_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, instruction_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute task weights from instruction embedding.
        
        Args:
            instruction_embedding: (batch_size, instruction_dim) instruction encoding
            
        Returns:
            (batch_size, num_tasks) task weight logits (before softmax)
        """
        logits = self.mlp(instruction_embedding)  # (batch_size, num_tasks)
        return logits
    
    def get_task_weights(self, instruction_embedding: torch.Tensor) -> torch.Tensor:
        """
        Get softmax-normalized task weights.
        
        Args:
            instruction_embedding: (batch_size, instruction_dim) instruction encoding
            
        Returns:
            (batch_size, num_tasks) task weights (probabilities sum to 1)
        """
        logits = self.forward(instruction_embedding)
        weights = F.softmax(logits, dim=-1)
        return weights


class TrajectoryLoRAAdapter(nn.Module):
    """
    Single LoRA adapter module: g_i(x_t, t, c) = A_i(c) * ReLU(B_i([h_t; Emb(t)])).
    
    This implements one low-rank perturbation module. Multiple modules can be combined.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        instruction_dim: int,
        activation: str = "relu",
        init_scale: float = 0.01,
    ):
        """
        Args:
            hidden_dim: Hidden dimension (d)
            rank: Low-rank dimension (r)
            instruction_dim: Dimension of instruction embedding
            activation: Activation function ("relu", "gelu", "silu")
            init_scale: Initialization scale for weights
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # Down-projection: B_i: R^d -> R^r
        # Input is [h_t; time_emb], so input dim is hidden_dim (we add time emb to h_t)
        self.down_proj = nn.Linear(hidden_dim, rank, bias=True)
        
        # Up-projection: A_i(c): R^r -> R^d
        # Base up-projection
        self.up_proj_base = nn.Linear(rank, hidden_dim, bias=False)
        
        # Instruction conditioning for up-projection
        # We implement A_i(c) = W_A + W_{A,cond} * Enc(c)
        # This is done via a modulation: A_i(c) v = W_A v + modulation(c, v)
        # We use FiLM-style conditioning: scale and shift
        self.up_proj_cond_scale = nn.Linear(instruction_dim, rank, bias=False)
        self.up_proj_cond_shift = nn.Linear(instruction_dim, hidden_dim, bias=False)
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj_base.weight, std=init_scale)
        nn.init.zeros_(self.up_proj_cond_scale.weight)
        nn.init.zeros_(self.up_proj_cond_shift.weight)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        instruction_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply LoRA perturbation.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) hidden states h_t
            instruction_embedding: (batch_size, instruction_dim) instruction encoding
            
        Returns:
            (batch_size, seq_len, hidden_dim) perturbation delta
        """
        # Down-project
        down = self.down_proj(hidden_states)  # (batch_size, seq_len, rank)
        
        # Apply activation
        down = self.activation(down)
        
        # Get instruction-conditioned scaling
        scale = self.up_proj_cond_scale(instruction_embedding)  # (batch_size, rank)
        scale = scale[:, None, :]  # (batch_size, 1, rank)
        
        # Apply scaling to down-projected features
        down_scaled = down * (1.0 + scale)  # FiLM-style modulation
        
        # Up-project (base)
        up = self.up_proj_base(down_scaled)  # (batch_size, seq_len, hidden_dim)
        
        # Add instruction-conditioned shift
        shift = self.up_proj_cond_shift(instruction_embedding)  # (batch_size, hidden_dim)
        shift = shift[:, None, :]  # (batch_size, 1, hidden_dim)
        up = up + shift
        
        return up


class LoRADiffusionModule(nn.Module):
    """
    Complete LoRA-Diffusion module with step-adaptive ranks and scaling.
    
    Manages multiple TrajectoryLoRAAdapter modules and applies step-adaptive
    configuration based on diffusion timestep.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self.hidden_dim = config["model"]["hidden_dim"]
        self.vocab_size = config["model"]["vocab_size"]
        self.num_diffusion_steps = config["diffusion"]["num_steps"]
        
        # LoRA configuration
        lora_config = config["lora"]
        self.rank_early = lora_config["rank_early"]
        self.rank_mid = lora_config["rank_mid"]
        self.rank_late = lora_config["rank_late"]
        self.scaling_early = lora_config["scaling_early"]
        self.scaling_mid = lora_config["scaling_mid"]
        self.scaling_late = lora_config["scaling_late"]
        self.num_modules = lora_config["num_modules"]
        
        # Instruction encoder
        instruction_hidden = lora_config["instruction_encoder_hidden"]
        instruction_layers = lora_config["instruction_encoder_layers"]
        self.instruction_encoder = InstructionEncoder(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            output_dim=instruction_hidden,
            num_layers=instruction_layers,
        )
        # Project instruction embedding to hidden_dim for base model conditioning
        self.instruction_to_hidden = nn.Linear(instruction_hidden, self.hidden_dim)
        nn.init.zeros_(self.instruction_to_hidden.weight)
        nn.init.zeros_(self.instruction_to_hidden.bias)
        
        # Create LoRA adapters for each phase and module
        # We create separate adapters for early/mid/late phases
        self.adapters_early = nn.ModuleList([
            TrajectoryLoRAAdapter(
                hidden_dim=self.hidden_dim,
                rank=self.rank_early,
                instruction_dim=instruction_hidden,
                activation=config.get("method", {}).get("activation", "relu"),
                init_scale=config.get("method", {}).get("init_scale", 0.01),
            )
            for _ in range(self.num_modules)
        ])
        
        self.adapters_mid = nn.ModuleList([
            TrajectoryLoRAAdapter(
                hidden_dim=self.hidden_dim,
                rank=self.rank_mid,
                instruction_dim=instruction_hidden,
                activation=config.get("method", {}).get("activation", "relu"),
                init_scale=config.get("method", {}).get("init_scale", 0.01),
            )
            for _ in range(self.num_modules)
        ])
        
        self.adapters_late = nn.ModuleList([
            TrajectoryLoRAAdapter(
                hidden_dim=self.hidden_dim,
                rank=self.rank_late,
                instruction_dim=instruction_hidden,
                activation=config.get("method", {}).get("activation", "relu"),
                init_scale=config.get("method", {}).get("init_scale", 0.01),
            )
            for _ in range(self.num_modules)
        ])
        
        # Regularization weights
        self.rank_reg_weight = lora_config["rank_reg_weight"]
        self.orth_reg_weight = lora_config["orth_reg_weight"]
    
    def get_step_config(self, timestep: int) -> tuple:
        """
        Get rank and scaling for a given timestep.
        
        Args:
            timestep: Diffusion timestep t
            
        Returns:
            (adapters, scaling) for this timestep
        """
        T = self.num_diffusion_steps
        
        if timestep > 2 * T // 3:  # Early phase
            return self.adapters_early, self.scaling_early
        elif timestep > T // 3:  # Middle phase
            return self.adapters_mid, self.scaling_mid
        else:  # Late phase
            return self.adapters_late, self.scaling_late
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        instruction_ids: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute trajectory perturbation.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) hidden states h_t
            timesteps: (batch_size,) diffusion timesteps
            instruction_ids: (batch_size, inst_len) instruction token IDs
            instruction_mask: (batch_size, inst_len) instruction attention mask
            
        Returns:
            (batch_size, seq_len, hidden_dim) trajectory perturbation delta
        """
        batch_size = hidden_states.size(0)
        
        # Encode instruction
        instruction_emb = self.instruction_encoder(
            instruction_ids,
            attention_mask=instruction_mask,
        )  # (batch_size, instruction_dim)
        
        # Compute perturbation for each sample based on its timestep
        # Note: In a batch, samples may have different timesteps
        # For efficiency, we group by phase
        delta = torch.zeros_like(hidden_states)
        
        # Process each phase
        for phase in ["early", "mid", "late"]:
            if phase == "early":
                mask = timesteps > 2 * self.num_diffusion_steps // 3
                adapters = self.adapters_early
                scaling = self.scaling_early
            elif phase == "mid":
                mask = (timesteps > self.num_diffusion_steps // 3) & (timesteps <= 2 * self.num_diffusion_steps // 3)
                adapters = self.adapters_mid
                scaling = self.scaling_mid
            else:  # late
                mask = timesteps <= self.num_diffusion_steps // 3
                adapters = self.adapters_late
                scaling = self.scaling_late
            
            if mask.any():
                # Get samples in this phase
                indices = mask.nonzero(as_tuple=True)[0]
                h_phase = hidden_states[indices]
                inst_emb_phase = instruction_emb[indices]
                
                # Apply all modules and sum
                phase_delta = torch.zeros_like(h_phase)
                for adapter in adapters:
                    phase_delta = phase_delta + adapter(h_phase, inst_emb_phase)
                
                # Apply scaling
                phase_delta = phase_delta * scaling
                
                # Assign back
                delta[indices] = phase_delta
        
        return delta
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization losses.
        
        Returns:
            Scalar regularization loss
        """
        total_loss = 0.0
        
        # Rank regularization: Frobenius norm as proxy for nuclear norm
        # ||W||_* ≈ ||W||_F when rank is low
        if self.rank_reg_weight > 0:
            rank_loss = 0.0
            for adapters in [self.adapters_early, self.adapters_mid, self.adapters_late]:
                for adapter in adapters:
                    rank_loss += torch.norm(adapter.down_proj.weight, p="fro") ** 2
                    rank_loss += torch.norm(adapter.up_proj_base.weight, p="fro") ** 2
            total_loss += self.rank_reg_weight * rank_loss
        
        # Orthogonality regularization: encourage different modules to be orthogonal
        if self.orth_reg_weight > 0 and self.num_modules > 1:
            orth_loss = 0.0
            for adapters in [self.adapters_early, self.adapters_mid, self.adapters_late]:
                # Compute pairwise inner products of up-projection matrices
                for i in range(len(adapters)):
                    for j in range(i + 1, len(adapters)):
                        W_i = adapters[i].up_proj_base.weight  # (hidden_dim, rank_i)
                        W_j = adapters[j].up_proj_base.weight  # (hidden_dim, rank_j)
                        # Inner product: ||W_i^T W_j||_F^2
                        inner_prod = torch.mm(W_i.t(), W_j)  # (rank_i, rank_j)
                        orth_loss += torch.norm(inner_prod, p="fro") ** 2
            total_loss += self.orth_reg_weight * orth_loss
        
        return total_loss
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters."""
        counts = {
            "instruction_encoder": sum(p.numel() for p in self.instruction_encoder.parameters()),
            "instruction_to_hidden": sum(p.numel() for p in self.instruction_to_hidden.parameters()),
            "adapters_early": sum(
                sum(p.numel() for p in adapter.parameters())
                for adapter in self.adapters_early
            ),
            "adapters_mid": sum(
                sum(p.numel() for p in adapter.parameters())
                for adapter in self.adapters_mid
            ),
            "adapters_late": sum(
                sum(p.numel() for p in adapter.parameters())
                for adapter in self.adapters_late
            ),
        }
        counts["total"] = sum(counts.values())
        return counts


class MultiTaskLoRAComposer(nn.Module):
    """
    Composes multiple task-specific LoRA modules for zero-shot task combination.
    
    Manages M task-specific LoRA modules {φ_i^(j)} and combines them using router weights.
    
    From paper Section 3.5, Equation 13:
    x_{t-1} = f_θ₀(x_t, t, c) + Σ_j w_j * Σ_i σ(t) * g_φ_i^(j)(x_t, t, c)
    
    where:
    - w_j = softmax(Router(Enc(c))) are task weights
    - g_φ_i^(j) are task-specific LoRA modules
    - σ(t) is step-adaptive scaling
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        task_names: List[str],
        instruction_encoder: Optional[InstructionEncoder] = None,
    ):
        """
        Args:
            config: Configuration dictionary
            task_names: List of task names (e.g., ["sst2", "squad", "xsum"])
            instruction_encoder: Shared instruction encoder (optional, creates new if None)
        """
        super().__init__()
        
        self.config = config
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        # Create task name to index mapping
        self.task_to_idx = {name: idx for idx, name in enumerate(task_names)}
        
        # Instruction encoder (shared across tasks)
        if instruction_encoder is None:
            lora_config = config["lora"]
            instruction_hidden = lora_config["instruction_encoder_hidden"]
            instruction_layers = lora_config["instruction_encoder_layers"]
            self.instruction_encoder = InstructionEncoder(
                vocab_size=config["model"]["vocab_size"],
                hidden_dim=config["model"]["hidden_dim"],
                output_dim=instruction_hidden,
                num_layers=instruction_layers,
            )
        else:
            self.instruction_encoder = instruction_encoder
        
        # Create router
        instruction_hidden = config["lora"]["instruction_encoder_hidden"]
        router_hidden = config.get("multi_task", {}).get("router_hidden_dim", 512)
        self.router = TaskRouter(
            instruction_dim=instruction_hidden,
            num_tasks=self.num_tasks,
            hidden_dim=router_hidden,
        )
        
        # Store task-specific LoRA modules (loaded separately via load_task_module)
        self.task_modules: Dict[str, LoRADiffusionModule] = {}
        
        # Composition mode: "router" (default), "uniform" (1/M), "task_arithmetic" (sum of deltas)
        self.composition_mode = "router"
        
        # Step-adaptive scaling (shared from config)
        lora_config = config["lora"]
        self.scaling_early = lora_config["scaling_early"]
        self.scaling_mid = lora_config["scaling_mid"]
        self.scaling_late = lora_config["scaling_late"]
        self.num_diffusion_steps = config["diffusion"]["num_steps"]
    
    def load_task_module(self, task_name: str, checkpoint_path: Path) -> None:
        """
        Load a task-specific LoRA module from checkpoint.
        
        Args:
            task_name: Name of the task (must be in task_names)
            checkpoint_path: Path to checkpoint directory containing lora_module.pt
        """
        if task_name not in self.task_names:
            raise ValueError(
                f"Task '{task_name}' not in task_names: {self.task_names}"
            )
        
        if task_name in self.task_modules:
            raise ValueError(f"Task module for '{task_name}' already loaded")
        
        # Load LoRA module (check checkpoint root then final_model/ subdir)
        lora_module = LoRADiffusionModule(self.config)
        lora_path = checkpoint_path / "lora_module.pt"
        if not lora_path.exists():
            lora_path = checkpoint_path / "final_model" / "lora_module.pt"
        if not lora_path.exists():
            raise FileNotFoundError(
                f"LoRA module not found at {checkpoint_path} (tried lora_module.pt and final_model/lora_module.pt). "
                f"Expected checkpoint directory with lora_module.pt"
            )
        
        state_dict = torch.load(lora_path, map_location="cpu")
        lora_module.load_state_dict(state_dict)
        lora_module.eval()  # Set to eval mode for inference
        
        # Freeze the loaded module
        for param in lora_module.parameters():
            param.requires_grad = False
        
        self.task_modules[task_name] = lora_module
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        instruction_ids: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute composed trajectory perturbation from multiple task-specific LoRA modules.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) hidden states h_t
            timesteps: (batch_size,) diffusion timesteps
            instruction_ids: (batch_size, inst_len) instruction token IDs
            instruction_mask: (batch_size, inst_len) instruction attention mask
            
        Returns:
            (batch_size, seq_len, hidden_dim) composed trajectory perturbation delta
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Encode instruction (shared encoder)
        instruction_emb = self.instruction_encoder(
            instruction_ids,
            attention_mask=instruction_mask,
        )  # (batch_size, instruction_dim)
        
        # Resolve task weights by composition mode
        mode = getattr(self, "composition_mode", "router")
        if mode == "uniform":
            # Average: w_j = 1/M
            task_weights = torch.ones(batch_size, self.num_tasks, device=device) / self.num_tasks
        elif mode == "task_arithmetic":
            # Task arithmetic: sum of deltas (weight 1 per task)
            task_weights = torch.ones(batch_size, self.num_tasks, device=device)
        else:
            # Default: router
            task_weights = self.router.get_task_weights(instruction_emb)  # (batch_size, num_tasks)
        
        # Initialize composed perturbation
        composed_delta = torch.zeros_like(hidden_states)
        
        # For each task, compute perturbation and weight
        for task_name, task_idx in self.task_to_idx.items():
            if task_name not in self.task_modules:
                continue
            
            lora_module = self.task_modules[task_name]
            task_delta = lora_module(
                hidden_states=hidden_states,
                timesteps=timesteps,
                instruction_ids=instruction_ids,
                instruction_mask=instruction_mask,
            )  # (batch_size, seq_len, hidden_dim)
            
            weight = task_weights[:, task_idx]  # (batch_size,)
            weight = weight[:, None, None]  # (batch_size, 1, 1)
            composed_delta = composed_delta + weight * task_delta
        
        return composed_delta
    
    def compute_router_loss(
        self,
        instruction_embedding: torch.Tensor,
        task_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute router training loss.
        
        From paper Equation 12:
        L_router = -log w_j where j is ground-truth task label
        
        Args:
            instruction_embedding: (batch_size, instruction_dim) instruction encoding
            task_labels: (batch_size,) ground-truth task labels (integer indices)
            
        Returns:
            Scalar router loss
        """
        router_logits = self.router(instruction_embedding)  # (batch_size, num_tasks)
        router_loss = F.cross_entropy(router_logits, task_labels)
        return router_loss
    
    def get_task_weights(
        self,
        instruction_ids: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get task weights for given instructions (for inspection/debugging).
        
        Args:
            instruction_ids: (batch_size, inst_len) instruction token IDs
            instruction_mask: (batch_size, inst_len) instruction attention mask
            
        Returns:
            (batch_size, num_tasks) task weights (probabilities)
        """
        instruction_emb = self.instruction_encoder(
            instruction_ids,
            attention_mask=instruction_mask,
        )
        return self.router.get_task_weights(instruction_emb)
