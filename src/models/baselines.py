"""Baseline Parameter-Efficient Fine-Tuning Methods.

Implements Weight LoRA, Prefix Tuning, Adapters, and BitFit for comparison.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightLoRAModule(nn.Module):
    """
    Standard Weight-based LoRA applied to attention projection matrices.
    
    Implements: W' = W + BA where W is frozen, B and A are learned.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout probability
        """
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA: output = Wx + (BA)x.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            LoRA output (..., out_features)
        """
        # Note: The base Wx is computed by the original layer
        # This module only computes the LoRA perturbation (BA)x
        lora_out = self.lora_B(self.dropout(self.lora_A(x)))
        return lora_out * self.scaling


def apply_weight_lora_to_model(
    model: nn.Module,
    rank: int = 64,
    alpha: float = 16.0,
    target_modules: list = ["query", "key", "value", "output"],
) -> int:
    """
    Apply Weight LoRA to specific modules in a transformer model.
    
    Args:
        model: Base model
        rank: LoRA rank
        alpha: LoRA scaling
        target_modules: List of module names to apply LoRA to
        
    Returns:
        Number of LoRA modules added
    """
    import math
    
    count = 0
    # Collect modules first to avoid recursion during modification
    modules_to_modify = []
    for name, module in model.named_modules():
        # Check if this is a target module
        is_target = any(target in name.lower() for target in target_modules)
        
        if is_target and isinstance(module, nn.Linear):
            modules_to_modify.append((name, module))
    
    # Now modify modules
    for name, module in modules_to_modify:
            # Get dimensions
            in_features = module.in_features
            out_features = module.out_features
            
            # Create LoRA module
            lora = WeightLoRAModule(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
            )
            
            # Wrap the original forward with LoRA
            original_forward = module.forward
            
            def make_forward_with_lora(orig_forward, lora_module):
                def forward_with_lora(x):
                    # Base output
                    base_out = orig_forward(x)
                    # LoRA perturbation
                    lora_out = lora_module(x)
                    return base_out + lora_out
                return forward_with_lora
            
            module.forward = make_forward_with_lora(original_forward, lora)
            
            # Register LoRA as a submodule
            module.lora_module = lora
            
            count += 1
    
    return count


class PrefixTuningModule(nn.Module):
    """
    Prefix Tuning: prepend learnable soft prompts to each layer.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_virtual_tokens: int,
        hidden_dim: int,
        num_heads: int,
        prefix_hidden_dim: int = 512,
        reparameterization: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_layers: Number of transformer layers
            num_virtual_tokens: Number of prefix tokens
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            prefix_hidden_dim: Hidden dim for prefix MLP reparameterization
            reparameterization: Whether to use MLP reparameterization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.reparameterization = reparameterization
        
        if reparameterization:
            # Use MLP to generate prefix
            # Input: learnable embedding, Output: key and value for each layer
            self.prefix_embedding = nn.Embedding(num_virtual_tokens, hidden_dim)
            self.prefix_mlp = nn.Sequential(
                nn.Linear(hidden_dim, prefix_hidden_dim),
                nn.Tanh(),
                nn.Linear(prefix_hidden_dim, num_layers * 2 * hidden_dim),
                nn.Dropout(dropout),
            )
        else:
            # Directly learn key and value for each layer
            # Shape: (num_layers, 2, num_virtual_tokens, hidden_dim)
            # 2 for key and value
            self.prefix_params = nn.Parameter(
                torch.randn(num_layers, 2, num_virtual_tokens, hidden_dim) * 0.01
            )
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate prefix key-value pairs for all layers.
        
        Args:
            batch_size: Batch size
            
        Returns:
            (num_layers, batch_size, 2, num_virtual_tokens, hidden_dim)
            prefix key-values for each layer
        """
        if self.reparameterization:
            # Generate from MLP
            indices = torch.arange(self.num_virtual_tokens, device=self.prefix_embedding.weight.device)
            prefix_emb = self.prefix_embedding(indices)  # (num_virtual_tokens, hidden_dim)
            prefix_flat = self.prefix_mlp(prefix_emb)  # (num_virtual_tokens, num_layers * 2 * hidden_dim)
            
            # Reshape to (num_layers, 2, num_virtual_tokens, hidden_dim)
            prefix_kv = prefix_flat.view(
                self.num_virtual_tokens,
                self.num_layers,
                2,
                self.hidden_dim,
            ).permute(1, 2, 0, 3)
            
            # Expand for batch
            prefix_kv = prefix_kv.unsqueeze(1).expand(-1, batch_size, -1, -1, -1)
        else:
            # Direct parameters
            prefix_kv = self.prefix_params.unsqueeze(1).expand(-1, batch_size, -1, -1, -1)
        
        return prefix_kv


class AdapterModule(nn.Module):
    """
    Adapter Layer: bottleneck feed-forward module.
    
    Inserts a down-projection -> activation -> up-projection after attention/FFN.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 256,
        non_linearity: str = "relu",
        dropout: float = 0.1,
        scaling: float = 1.0,
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            bottleneck_dim: Adapter bottleneck dimension
            non_linearity: Activation function
            dropout: Dropout probability
            scaling: Output scaling factor
        """
        super().__init__()
        
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = scaling
        
        # Activation
        if non_linearity == "relu":
            self.activation = nn.ReLU()
        elif non_linearity == "gelu":
            self.activation = nn.GELU()
        elif non_linearity == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {non_linearity}")
        
        # Initialize
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter.
        
        Args:
            hidden_states: (..., hidden_dim)
            
        Returns:
            (..., hidden_dim) adapter output
        """
        down = self.down_proj(hidden_states)
        down = self.activation(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        return up * self.scaling


def apply_adapters_to_model(
    model: nn.Module,
    bottleneck_dim: int = 256,
    insert_after: list = ["attention", "ffn"],
) -> int:
    """
    Add adapter layers to model.
    
    Args:
        model: Base model
        bottleneck_dim: Adapter bottleneck dimension
        insert_after: Where to insert adapters
        
    Returns:
        Number of adapters added
    """
    # This is a simplified implementation
    # In practice, would need model-specific insertion logic
    count = 0
    
    # Get hidden_dim from model config
    if hasattr(model, "hidden_dim"):
        hidden_dim = model.hidden_dim
    elif hasattr(model, "config") and "hidden_dim" in model.config.get("model", {}):
        hidden_dim = model.config["model"]["hidden_dim"]
    else:
        hidden_dim = 768  # Default BERT hidden_dim
    
    # Collect modules first to avoid recursion during modification
    modules_to_modify = []
    for name, module in model.named_modules():
        # Look for BERT encoder layers - insert adapters after the layer output
        is_bert_layer = "transformer.encoder.layer" in name and name.endswith("layer")
        is_attention_output = "attention.output" in name.lower()
        is_ffn_output = "output" in name.lower() and "intermediate" not in name.lower() and "attention" not in name.lower()
        
        should_insert = False
        actual_dim = hidden_dim
        
        if "ffn" in insert_after:
            # Insert after FFN output (BertLayer.output)
            if is_ffn_output and "layer" in name:
                should_insert = True
                actual_dim = hidden_dim  # Layer output is hidden_dim
        elif "attention" in insert_after:
            # Insert after attention output
            if is_attention_output:
                should_insert = True
                actual_dim = hidden_dim  # Attention output is hidden_dim
        
        if should_insert and hasattr(module, "forward") and not hasattr(module, "adapter_module"):
            modules_to_modify.append((name, module, actual_dim))
    
    # Now modify modules
    for name, module, actual_dim in modules_to_modify:
                adapter = AdapterModule(
                    hidden_dim=actual_dim,  # Use actual dimension
                    bottleneck_dim=bottleneck_dim,
                )
                
                # Wrap forward
                original_forward = module.forward
                
                def make_forward_with_adapter(orig_forward, adapter_module):
                    def forward_with_adapter(*args, **kwargs):
                        output = orig_forward(*args, **kwargs)
                        # Handle different output formats
                        if isinstance(output, tuple):
                            hidden_states = output[0]
                            adapter_out = adapter_module(hidden_states)
                            return (hidden_states + adapter_out,) + output[1:]
                        else:
                            return output + adapter_module(output)
                    return forward_with_adapter
                
                wrapped_forward = make_forward_with_adapter(original_forward, adapter)
                module.forward = wrapped_forward
                module.adapter_module = adapter
                count += 1
    
    return count


def setup_bitfit(model: nn.Module, train_layer_norm: bool = True) -> int:
    """
    Setup BitFit: freeze all parameters except biases.
    
    Args:
        model: Model to configure
        train_layer_norm: Whether to train LayerNorm parameters
        
    Returns:
        Number of trainable parameters
    """
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # Freeze by default
        param.requires_grad = False
        
        # Enable gradient for biases
        if "bias" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        
        # Optionally enable LayerNorm
        if train_layer_norm and ("layer_norm" in name.lower() or "layernorm" in name.lower()):
            param.requires_grad = True
            trainable_count += param.numel()
    
    return trainable_count


import math  # Add this at the top for WeightLoRAModule
