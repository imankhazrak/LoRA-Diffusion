# Figures Directory

This directory contains placeholder references for figures that need to be generated from experimental outputs.

## Required Figures

1. **rank_ablation.png/pdf**: Rank vs. performance (left) and vs. trainable parameters (right). Step-adaptive ranks (8/32/64) achieve the best tradeoff.

2. **effective_rank.png/pdf**: Effective rank of LoRA modules across diffusion steps. Early steps exhibit higher effective rank.

3. **data_efficiency.png/pdf**: Validation accuracy vs. training data size (real SST-2 sweep). **Requires running the data-efficiency sweep first:** `python scripts/run_data_efficiency_sweep.py --output_dir ./outputs/data_efficiency_sweep` (writes `data_efficiency_results.json`). Then `python scripts/generate_figures.py` plots from that file.

4. **trajectory_visualization.png/pdf**: t-SNE visualization of denoising trajectories showing task-specific clusters.

5. **reg_ablation.png/pdf**: Regularizer ablation (job 44066468). Val acc. and train loss vs. regularization configuration. Run `python scripts/generate_figures.py` to generate from job 44066468 results.

## Generation

- **Data efficiency (Figure 3):** Run `scripts/run_data_efficiency_sweep.py` to train LoRA-Diffusion and weight LoRA at 10%, 20%, 40%, 60%, 80%, 100% of SST-2 training data and collect validation accuracy; then run `scripts/generate_figures.py`.
- Other figures: `notebooks/analyze_results.ipynb` or `scripts/generate_figures.py`.

The figures can be in PNG, PDF, or other formats supported by LaTeX's `\includegraphics`.
