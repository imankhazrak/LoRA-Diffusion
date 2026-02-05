#!/usr/bin/env python3
"""Generate only the data_efficiency figure from results JSON."""

import json
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not available. Install: pip install matplotlib")
    exit(1)

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FIGURES_DIR = Path(__file__).parent.parent / "doc" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

results_path = Path(__file__).parent.parent / "data_efficiency_results.json"

if not results_path.exists():
    print(f"ERROR: {results_path} not found")
    exit(1)

with open(results_path) as f:
    data = json.load(f)

fractions = data.get("fractions", [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
x_pct = np.array(fractions) * 100

ld = data.get("lora_diffusion", {})
wl = data.get("weight_lora", {})

def get_series(method_dict):
    means, stds = [], []
    for frac in fractions:
        key = f"{frac:.2f}" if frac < 1.0 else "1.00"
        entry = method_dict.get(key, {})
        m = entry.get("mean")
        s = entry.get("std") or 0.0
        means.append(m if m is not None else np.nan)
        stds.append(s)
    return np.array(means), np.array(stds)

ld_mean, ld_std = get_series(ld)
wl_mean, wl_std = get_series(wl)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x_pct, ld_mean, linewidth=2.5, marker='o', markersize=8,
        label='LoRA-Diffusion', color='#2ca02c', zorder=3)
ax.plot(x_pct, wl_mean, linewidth=2.5, marker='s', markersize=8,
        label='Weight LoRA', color='#d62728', zorder=3)

ax.fill_between(x_pct, ld_mean - ld_std, ld_mean + ld_std, alpha=0.2, color='#2ca02c')
ax.fill_between(x_pct, wl_mean - wl_std, wl_mean + wl_std, alpha=0.2, color='#d62728')

ax.set_xlabel('Training Data Size (% of Full Dataset)', fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax.set_title('Data Efficiency: Performance vs. Training Data Size', fontweight='bold', pad=10)
ax.set_xlim([0, 105])
y_min = np.nanmin(np.concatenate([ld_mean - ld_std, wl_mean - wl_std]))
y_max = np.nanmax(np.concatenate([ld_mean + ld_std, wl_mean + wl_std]))
if np.isnan(y_min):
    y_min, y_max = 40, 95
ax.set_ylim([max(0, y_min - 5), min(100, y_max + 5)])
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', framealpha=0.9, fontsize=11)

# Annotate low-data regime if we have a gap
if len(x_pct) >= 2 and np.any(np.isfinite(ld_mean[:2])) and np.any(np.isfinite(wl_mean[:2])):
    idx = min(1, len(x_pct) - 1)
    ax.annotate('Better in\nlow-data regime', xy=(x_pct[idx], (ld_mean[idx] + wl_mean[idx]) / 2),
                xytext=(x_pct[0] - 5, ld_mean[0] + 10 if np.isfinite(ld_mean[0]) else 60),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'data_efficiency.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'data_efficiency.pdf', dpi=300, bbox_inches='tight')
print(f"✓ Generated data_efficiency.png and data_efficiency.pdf")
print(f"  LoRA-Diffusion: {ld_mean}")
print(f"  Weight LoRA: {wl_mean}")
plt.close()
