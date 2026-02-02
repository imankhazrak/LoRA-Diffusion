#!/usr/bin/env python3
"""Analyze effective rank of LoRA-Diffusion modules."""
import argparse
import json
import math
from pathlib import Path
import torch
import numpy as np
def effective_rank_from_singular_values(s, eps=1e-10):
    """Effective rank = exp(entropy of normalized singular values)."""
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > eps]
    entropy = -(p * (p + eps).log()).sum().item()
    return math.exp(min(entropy, math.log(s.numel() + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--output", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not args.checkpoint or not Path(args.checkpoint).exists():
        r, d = 8, 768
        B = torch.randn(d, r) / (r ** 0.5)
        A = torch.randn(r, d) / (r ** 0.5)
        s = torch.linalg.svdvals(A @ B)
        r_eff = effective_rank_from_singular_values(s)
        out = {"effective_rank_random_r8": r_eff}
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
        print("Effective rank:", r_eff)
        return
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    ranks = []
    for k, p in ckpt.items():
        if p.dim() >= 2:
            s = torch.linalg.svdvals(p.detach().float())
            ranks.append(effective_rank_from_singular_values(s))
    out = {"mean_effective_rank": float(np.mean(ranks))} if ranks else {}
    if args.output and out:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
if __name__ == "__main__":
    main()
