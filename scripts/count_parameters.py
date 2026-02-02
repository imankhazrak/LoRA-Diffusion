#!/usr/bin/env python3
"""Count trainable parameters and storage sizes for all PEFT methods."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MaskedDiffusionTransformer, LoRADiffusionModule
from src.models.baselines import (
    apply_weight_lora_to_model,
    setup_bitfit,
    apply_adapters_to_model,
    PrefixTuningModule,
)
from src.utils import load_config


def count_model_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_storage_size_mb(model_state_dict):
    """Estimate storage size in MB (assuming float32)."""
    total_params = sum(p.numel() for p in model_state_dict.values())
    # 4 bytes per float32 parameter
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def count_lora_diffusion(config):
    """Count parameters for LoRA-Diffusion."""
    model = MaskedDiffusionTransformer(config)
    lora_module = LoRADiffusionModule(config)
    
    base_total, _ = count_model_parameters(model)
    lora_total, lora_trainable = count_model_parameters(lora_module)
    
    # Estimate storage
    lora_state = lora_module.state_dict()
    storage_mb = estimate_storage_size_mb(lora_state)
    
    return {
        "method": "lora_diffusion",
        "base_params": base_total,
        "trainable_params": lora_trainable,
        "trainable_percent": (lora_trainable / base_total) * 100,
        "storage_mb": storage_mb,
        "breakdown": lora_module.count_parameters(),
    }


def count_weight_lora(config):
    """Count parameters for Weight LoRA."""
    model = MaskedDiffusionTransformer(config)
    
    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA
    lora_config = config.get("lora", {})
    rank = lora_config.get("rank", 64)
    alpha = lora_config.get("alpha", 16)
    target_modules = lora_config.get("target_modules", ["query", "key", "value", "output"])
    
    apply_weight_lora_to_model(model, rank=rank, alpha=alpha, target_modules=target_modules)
    
    base_total, _ = count_model_parameters(model)
    _, trainable = count_model_parameters(model)
    
    # Estimate storage (only LoRA modules)
    lora_state = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_module"):
            lora_state[f"{name}.lora_A.weight"] = module.lora_module.lora_A.weight
            lora_state[f"{name}.lora_B.weight"] = module.lora_module.lora_B.weight
    
    storage_mb = estimate_storage_size_mb(lora_state) if lora_state else 0.0
    
    return {
        "method": "weight_lora",
        "base_params": base_total,
        "trainable_params": trainable,
        "trainable_percent": (trainable / base_total) * 100,
        "storage_mb": storage_mb,
    }


def count_adapters(config):
    """Count parameters for Adapter layers."""
    model = MaskedDiffusionTransformer(config)
    
    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply adapters
    adapter_config = config.get("adapter", {})
    bottleneck_dim = adapter_config.get("bottleneck_dim", 256)
    insert_after = adapter_config.get("insert_after", ["attention", "ffn"])
    
    apply_adapters_to_model(model, bottleneck_dim=bottleneck_dim, insert_after=insert_after)
    
    base_total, _ = count_model_parameters(model)
    _, trainable = count_model_parameters(model)
    
    # Estimate storage (only adapter modules)
    adapter_state = {}
    for name, module in model.named_modules():
        if hasattr(module, "adapter_module"):
            adapter_state[f"{name}.down_proj.weight"] = module.adapter_module.down_proj.weight
            adapter_state[f"{name}.down_proj.bias"] = module.adapter_module.down_proj.bias
            adapter_state[f"{name}.up_proj.weight"] = module.adapter_module.up_proj.weight
            adapter_state[f"{name}.up_proj.bias"] = module.adapter_module.up_proj.bias
    
    storage_mb = estimate_storage_size_mb(adapter_state) if adapter_state else 0.0
    
    return {
        "method": "adapters",
        "base_params": base_total,
        "trainable_params": trainable,
        "trainable_percent": (trainable / base_total) * 100,
        "storage_mb": storage_mb,
    }


def count_bitfit(config):
    """Count parameters for BitFit."""
    model = MaskedDiffusionTransformer(config)
    
    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
    
    # Setup BitFit
    bitfit_config = config.get("bitfit", {})
    train_layer_norm = bitfit_config.get("train_layer_norm", True)
    
    setup_bitfit(model, train_layer_norm=train_layer_norm)
    
    base_total, _ = count_model_parameters(model)
    _, trainable = count_model_parameters(model)
    
    # Estimate storage (only biases)
    bias_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            bias_state[name] = param
    
    storage_mb = estimate_storage_size_mb(bias_state) if bias_state else 0.0
    
    return {
        "method": "bitfit",
        "base_params": base_total,
        "trainable_params": trainable,
        "trainable_percent": (trainable / base_total) * 100,
        "storage_mb": storage_mb,
    }


def count_prefix_tuning(config):
    """Count parameters for Prefix Tuning."""
    model = MaskedDiffusionTransformer(config)
    
    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
    
    # Create prefix module
    prefix_config = config.get("prefix", {})
    num_virtual_tokens = prefix_config.get("num_virtual_tokens", 32)
    prefix_hidden_dim = prefix_config.get("prefix_hidden_dim", 512)
    reparameterization = prefix_config.get("reparameterization", True)
    
    prefix_module = PrefixTuningModule(
        num_layers=model.num_layers,
        num_virtual_tokens=num_virtual_tokens,
        hidden_dim=model.hidden_dim,
        num_heads=model.num_heads,
        prefix_hidden_dim=prefix_hidden_dim,
        reparameterization=reparameterization,
        dropout=prefix_config.get("prefix_dropout", 0.1),
    )
    
    base_total, _ = count_model_parameters(model)
    prefix_total, prefix_trainable = count_model_parameters(prefix_module)
    
    # Estimate storage
    prefix_state = prefix_module.state_dict()
    storage_mb = estimate_storage_size_mb(prefix_state)
    
    return {
        "method": "prefix_tuning",
        "base_params": base_total,
        "trainable_params": prefix_trainable,
        "trainable_percent": (prefix_trainable / base_total) * 100,
        "storage_mb": storage_mb,
    }


def count_full_finetuning(config):
    """Count parameters for full fine-tuning."""
    model = MaskedDiffusionTransformer(config)
    
    base_total, base_trainable = count_model_parameters(model)
    
    # Full model storage
    model_state = model.state_dict()
    storage_mb = estimate_storage_size_mb(model_state)
    
    return {
        "method": "full_ft",
        "base_params": base_total,
        "trainable_params": base_trainable,
        "trainable_percent": 100.0,
        "storage_mb": storage_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Count parameters for all PEFT methods")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Base config file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sst2",
        help="Task name (for task-specific configs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit", "prefix_tuning"],
        help="Methods to count",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(
        Path(args.config),
        task_name=args.task,
        method_name=None,  # We'll load method configs separately
    )
    
    results = {}
    
    # Count each method
    for method in args.methods:
        print(f"Counting parameters for {method}...")
        
        # Load method-specific config
        if method != "full_ft":
            method_config = load_config(
                Path(args.config),
                task_name=args.task,
                method_name=method,
            )
        else:
            method_config = config
        
        try:
            if method == "full_ft":
                counts = count_full_finetuning(method_config)
            elif method == "lora_diffusion":
                counts = count_lora_diffusion(method_config)
            elif method == "weight_lora":
                counts = count_weight_lora(method_config)
            elif method == "adapters":
                counts = count_adapters(method_config)
            elif method == "bitfit":
                counts = count_bitfit(method_config)
            elif method == "prefix_tuning":
                counts = count_prefix_tuning(method_config)
            else:
                print(f"Unknown method: {method}")
                continue
            
            results[method] = counts
            
            print(f"  Base params: {counts['base_params']:,}")
            print(f"  Trainable params: {counts['trainable_params']:,}")
            print(f"  Trainable %: {counts['trainable_percent']:.2f}%")
            print(f"  Storage: {counts['storage_mb']:.2f} MB")
            if "breakdown" in counts:
                print(f"  Breakdown: {counts['breakdown']}")
            print()
            
        except Exception as e:
            print(f"  Error counting {method}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Trainable Params':<20} {'Trainable %':<15} {'Storage (MB)':<15}")
    print("-" * 80)
    for method, counts in results.items():
        print(
            f"{method:<20} "
            f"{counts['trainable_params']:>15,} "
            f"{counts['trainable_percent']:>12.2f}% "
            f"{counts['storage_mb']:>12.2f}"
        )


if __name__ == "__main__":
    main()
