"""Loss functions for diffusion model training."""

from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_diffusion_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    lora_module: Optional[nn.Module] = None,
    config: Optional[Dict] = None,
    router: Optional[nn.Module] = None,
    task_labels: Optional[torch.Tensor] = None,
    instruction_encoder: Optional[nn.Module] = None,
    instruction_to_hidden: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute diffusion training loss.
    
    Args:
        model: Base diffusion model
        batch: Batch of data
        lora_module: Optional LoRA-Diffusion module
        config: Configuration dict
        
    Returns:
        loss: Scalar loss tensor
        metrics: Dictionary of metrics
    """
    device = next(model.parameters()).device
    
    # Get data
    target_ids = batch["target_ids"].to(device)
    target_mask = batch["target_mask"].to(device)
    instruction_ids = batch["instruction_ids"].to(device)
    instruction_mask = batch["instruction_mask"].to(device)
    
    # Get task labels if provided
    if task_labels is not None:
        task_labels = task_labels.to(device)
    
    batch_size = target_ids.size(0)
    
    # Sample random timesteps
    num_diffusion_steps = model.num_diffusion_steps
    timesteps = torch.randint(
        0,
        num_diffusion_steps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    
    # Apply forward diffusion to get noisy inputs
    xt, mask = model.forward_diffusion(target_ids, timesteps)
    
    if lora_module is not None:
        # Check if this is prefix tuning (different handling)
        if hasattr(lora_module, "num_virtual_tokens"):
            # Prefix tuning: prefixes are handled in model forward
            instruction_emb_for_base = None
        else:
            # LoRA-Diffusion: compute base output + trajectory perturbation
            
            # Get instruction embedding (for adapters)
            instruction_emb = lora_module.instruction_encoder(
                instruction_ids,
                attention_mask=instruction_mask,
            )
            # Project to hidden_dim for base model conditioning
            instruction_emb_for_base = lora_module.instruction_to_hidden(instruction_emb)
        
        # Check if prefix tuning
        if hasattr(lora_module, "num_virtual_tokens"):
            # Prefix tuning: prefixes are prepended to attention keys/values
            # For now, use standard forward (prefix integration would require model modification)
            # This is a simplified version - full prefix tuning needs attention modification
            logits = model.forward(
                input_ids=xt,
                timesteps=timesteps,
                attention_mask=target_mask,
                instruction_embedding=None,
            )
            # Prefix module parameters are trainable, so they'll be included in optimizer
            reg_loss = torch.tensor(0.0, device=device, requires_grad=False)
        else:
            # LoRA-Diffusion: compute base output + trajectory perturbation
            
            # Get base model output (frozen)
            with torch.no_grad():
                base_logits = model.forward(
                    input_ids=xt,
                    timesteps=timesteps,
                    attention_mask=target_mask,
                    instruction_embedding=instruction_emb_for_base,
                )
            
            # Get hidden representations for LoRA
            hidden_states = model.get_representation(
                input_ids=xt,
                timesteps=timesteps,
                attention_mask=target_mask,
            )
            
            # Compute LoRA perturbation
            delta = lora_module(
                hidden_states=hidden_states,
                timesteps=timesteps,
                instruction_ids=instruction_ids,
                instruction_mask=instruction_mask,
            )
            
            # Add perturbation to hidden states and get final logits
            perturbed_hidden = hidden_states + delta
            logits = model.output_head(perturbed_hidden)
            
            # Compute regularization loss
            reg_loss = lora_module.compute_regularization_loss()
        
        # Compute router loss if router is provided
        router_loss = 0.0
        if router is not None and task_labels is not None:
            # Get instruction embedding (already computed above)
            # instruction_emb is available from lora_module.instruction_encoder call
            router_logits = router(instruction_emb)
            router_loss = F.cross_entropy(router_logits, task_labels)
    else:
        # Baseline methods (full_ft, weight_lora, adapters, bitfit): use shared instruction encoder when provided
        instruction_emb = None
        if instruction_encoder is not None and instruction_to_hidden is not None:
            instruction_emb = instruction_encoder(
                instruction_ids,
                attention_mask=instruction_mask,
            )
            instruction_emb_for_base = instruction_to_hidden(instruction_emb)
        else:
            instruction_emb_for_base = None
        logits = model.forward(
            input_ids=xt,
            timesteps=timesteps,
            attention_mask=target_mask,
            instruction_embedding=instruction_emb_for_base,
        )
        reg_loss = 0.0
        router_loss = 0.0
        if router is not None and task_labels is not None and instruction_emb is not None:
            router_logits = router(instruction_emb)
            router_loss = F.cross_entropy(router_logits, task_labels)
    
    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none",
    )
    ce_loss = ce_loss.reshape(target_ids.shape)
    
    # Apply mask: only compute loss on valid and corrupted tokens
    valid_mask = target_mask.bool() & mask
    ce_loss = (ce_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
    
    # Total loss
    router_loss_weight = config.get("multi_task", {}).get("router_loss_weight", 1.0) if config else 1.0
    total_loss = ce_loss + reg_loss + router_loss_weight * router_loss
    
    # Compute accuracy
    with torch.no_grad():
        pred_tokens = logits.argmax(dim=-1)
        correct = (pred_tokens == target_ids) & valid_mask
        accuracy = correct.float().sum() / valid_mask.float().sum()
    
    metrics = {
        "loss": total_loss.item(),
        "ce_loss": ce_loss.item(),
        "reg_loss": reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
        "router_loss": router_loss.item() if isinstance(router_loss, torch.Tensor) else router_loss,
        "accuracy": accuracy.item(),
    }
    
    return total_loss, metrics
