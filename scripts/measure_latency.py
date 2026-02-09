#!/usr/bin/env python3
"""Measure inference latency (ms/example) with fixed batch size and diffusion steps T."""

import argparse
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MaskedDiffusionTransformer, LoRADiffusionModule
from src.models.lora_modules import InstructionEncoder
from src.utils import load_config, set_seed


def load_model_and_encoder(checkpoint_path: Path, config: dict, device: str):
    """Load model and optional lora/instruction encoder from checkpoint."""
    set_seed(config.get("training", {}).get("seed", 42))
    model = MaskedDiffusionTransformer(config).to(device)
    model.eval()
    
    # Load base model if present
    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    
    lora_module = None
    instruction_encoder = None
    instruction_to_hidden = None
    
    lora_path = checkpoint_path / "lora_module.pt"
    if lora_path.exists():
        lora_module = LoRADiffusionModule(config).to(device)
        lora_module.load_state_dict(torch.load(lora_path, map_location=device))
        lora_module.eval()
    
    enc_path = checkpoint_path / "instruction_encoder.pt"
    if enc_path.exists() and lora_module is None:
        lora_cfg = config.get("lora", {})
        model_cfg = config.get("model", {})
        instruction_encoder = InstructionEncoder(
            vocab_size=model_cfg.get("vocab_size", 30522),
            hidden_dim=model_cfg.get("hidden_dim", 768),
            output_dim=lora_cfg.get("instruction_encoder_hidden", 256),
            num_layers=lora_cfg.get("instruction_encoder_layers", 2),
        ).to(device)
        instruction_to_hidden = torch.nn.Linear(
            lora_cfg.get("instruction_encoder_hidden", 256),
            model_cfg.get("hidden_dim", 768),
        ).to(device)
        enc_data = torch.load(enc_path, map_location=device)
        instruction_encoder.load_state_dict(enc_data["instruction_encoder"])
        instruction_to_hidden.load_state_dict(enc_data["instruction_to_hidden"])
        instruction_encoder.eval()
        instruction_to_hidden.eval()
    
    return model, lora_module, instruction_encoder, instruction_to_hidden


def main():
    parser = argparse.ArgumentParser(description="Measure inference latency (ms/example)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Fixed batch size")
    parser.add_argument("--seq_len", type=int, default=5, help="Sequence length (e.g. classification)")
    parser.add_argument("--num_steps", type=int, default=100, help="Diffusion steps T")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs before timing")
    parser.add_argument("--runs", type=int, default=20, help="Number of timed runs")
    parser.add_argument("--output", type=str, default=None, help="Write ms_per_example to this JSON (or checkpoint dir)")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load config
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        summary_path = checkpoint_path / "training_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                config = json.load(f).get("config", {})
        else:
            raise FileNotFoundError(f"No config at {config_path} or {summary_path}")
    else:
        with open(config_path) as f:
            config = json.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, lora_module, instruction_encoder, instruction_to_hidden = load_model_and_encoder(
        checkpoint_path, config, device
    )
    
    batch_size = args.batch_size
    seq_len = args.seq_len
    T = getattr(model, "num_diffusion_steps", args.num_steps)
    
    # Dummy instruction embedding (batch_size, hidden_dim)
    hidden_dim = config.get("model", {}).get("hidden_dim", 768)
    instruction_emb = torch.zeros(batch_size, hidden_dim, device=device)
    
    @torch.no_grad()
    def run_one():
        if lora_module is not None:
            instruction_emb_out = instruction_emb
        elif instruction_encoder is not None:
            # We don't have real instruction ids here; use zeros (shape only matters for timing)
            instruction_emb_out = instruction_emb
        else:
            instruction_emb_out = None
        _ = model.sample(
            batch_size=batch_size,
            seq_len=seq_len,
            instruction_embedding=instruction_emb_out,
            device=device,
            greedy=True,
        )
    
    # Warmup
    for _ in range(args.warmup):
        run_one()
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(args.runs):
        run_one()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    total_examples = args.runs * batch_size
    ms_per_example = (elapsed / total_examples) * 1000.0
    
    result = {
        "ms_per_example": ms_per_example,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "diffusion_steps": T,
        "runs": args.runs,
        "total_seconds": elapsed,
    }
    
    out_path = Path(args.output) if args.output else checkpoint_path / "latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Latency: {ms_per_example:.2f} ms/example (batch_size={batch_size}, T={T})")
    print(f"Saved to {out_path}")
    return result


if __name__ == "__main__":
    main()
