# Figures Directory

This directory contains placeholder references for figures that need to be generated from experimental outputs.

## Required Figures

1. **rank_ablation.png/pdf**: Rank vs. performance (left) and vs. trainable parameters (right). Step-adaptive ranks (8/32/64) achieve the best tradeoff.

2. **effective_rank.png/pdf**: Effective rank of LoRA modules across diffusion steps. Early steps exhibit higher effective rank.

3. **data_efficiency.png/pdf**: Performance vs. training data size. LoRA-Diffusion shows better data efficiency than weight LoRA.

4. **trajectory_visualization.png/pdf**: t-SNE visualization of denoising trajectories showing task-specific clusters.

## Generation

These figures should be generated from experimental outputs using:
- `notebooks/analyze_results.ipynb`
- Or similar analysis scripts

The figures can be in PNG, PDF, or other formats supported by LaTeX's `\includegraphics`.
