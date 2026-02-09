"""Parameter accounting: two views (PEFT-only vs total trainable) for fairness reporting."""

from typing import Dict, Any, Optional
import torch.nn as nn


def count_params(module: nn.Module, only_trainable: bool = False) -> int:
    """Count parameters in a module."""
    if module is None:
        return 0
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def get_parameter_accounting(
    model: nn.Module,
    lora_module: Optional[nn.Module] = None,
    instruction_encoder: Optional[nn.Module] = None,
    instruction_to_hidden: Optional[nn.Module] = None,
    encoder_frozen: bool = True,
) -> Dict[str, Any]:
    """
    Compute parameter counts for two views (Instruction2.md).
    
    View 1 (PEFT-only): adaptation modules only; instruction encoder excluded.
    View 2 (Total trainable): PEFT + instruction encoder if encoder is trainable.
    
    Returns:
        total_params: base model total
        trainable_params_peft_only: trainable params excluding instruction encoder
        trainable_params_total: trainable params including encoder if not frozen
        breakdown: dict with base, instruction_encoder, trajectory/lora/adapters/bitfit
    """
    total_params = count_params(model, only_trainable=False)
    base_trainable = count_params(model, only_trainable=True)
    
    breakdown = {"base": total_params}
    encoder_params = 0
    if instruction_encoder is not None:
        encoder_params += count_params(instruction_encoder, only_trainable=False)
    if instruction_to_hidden is not None:
        encoder_params += count_params(instruction_to_hidden, only_trainable=False)
    breakdown["instruction_encoder"] = encoder_params
    
    if lora_module is not None:
        if hasattr(lora_module, "count_parameters"):
            lora_breakdown = lora_module.count_parameters()
            for k, v in lora_breakdown.items():
                if k != "total":
                    breakdown[f"lora_{k}"] = v
        # When encoder_frozen, trainable params in lora_module already exclude encoder (requires_grad=False)
        trainable_in_lora = count_params(lora_module, only_trainable=True)
        trainable_params_peft_only = trainable_in_lora
        trainable_params_total = trainable_in_lora
    else:
        # Baselines: PEFT = base trainable; total = base + encoder if not frozen
        trainable_params_peft_only = base_trainable
        enc_trainable = (count_params(instruction_encoder, only_trainable=True) if instruction_encoder else 0) + (
            count_params(instruction_to_hidden, only_trainable=True) if instruction_to_hidden else 0
        )
        trainable_params_total = base_trainable + (enc_trainable if not encoder_frozen else 0)
    
    return {
        "total_params": total_params,
        "trainable_params_peft_only": trainable_params_peft_only,
        "trainable_params_total": trainable_params_total,
        "breakdown": breakdown,
        "encoder_frozen": encoder_frozen,
    }
