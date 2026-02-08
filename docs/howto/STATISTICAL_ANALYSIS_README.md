# Multi-Seed Statistical Analysis Guide

This document explains how to run experiments with multiple seeds and perform comprehensive statistical analysis for the LoRA-Diffusion paper.

## Overview

All experiments are run with **10 independent random seeds (42-51)** to ensure robust and reproducible results. We compute:
- **Descriptive statistics**: mean, standard deviation, variance, 95% confidence intervals
- **Significance testing**: paired t-tests with p-values and Bonferroni correction
- **Effect sizes**: Cohen's d for practical significance
- **Robustness checks**: Shapiro-Wilk (normality), Levene (homogeneity of variance), Wilcoxon (non-parametric)

## Quick Start

### Option 1: Run Complete Pipeline (Automated)

```bash
# Run all experiments, evaluations, and generate tables
bash scripts/run_multi_seed_experiments.sh
```

This script will:
1. Train all methods with 10 seeds (50 experiments total)
2. Run generation-based evaluation on all checkpoints
3. Run classification-head evaluation on all checkpoints
4. Collect results and compute statistics
5. Generate LaTeX tables with mean±std and significance markers

### Option 2: Manual Step-by-Step

#### Step 1: Run Training Experiments

```bash
# Run all methods with 10 seeds
python scripts/run_experiments.py \
  --tasks sst2 \
  --methods full_ft lora_diffusion weight_lora adapters bitfit \
  --num_seeds 10 \
  --start_seed 42 \
  --output_dir ./outputs/multi_seed_experiments \
  --config configs/base_config.yaml
```

Or specify seeds explicitly:

```bash
python scripts/run_experiments.py \
  --tasks sst2 \
  --methods full_ft lora_diffusion weight_lora adapters bitfit \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --output_dir ./outputs/multi_seed_experiments
```

#### Step 2: Run Generation-Based Evaluation

```bash
for seed in {42..51}; do
  for method in full_ft lora_diffusion weight_lora adapters bitfit; do
    python scripts/evaluate.py \
      --checkpoint outputs/multi_seed_experiments/sst2_${method}_seed${seed} \
      --task sst2 \
      --split validation \
      --output_file outputs/multi_seed_experiments/sst2_${method}_seed${seed}/eval_results.json
  done
done
```

#### Step 3: Run Classification-Head Evaluation

```bash
for seed in {42..51}; do
  for method in full_ft lora_diffusion weight_lora adapters bitfit; do
    python scripts/evaluate.py \
      --checkpoint outputs/multi_seed_experiments/sst2_${method}_seed${seed} \
      --task sst2 \
      --split validation \
      --eval_classification_head \
      --output_file outputs/multi_seed_experiments/sst2_${method}_seed${seed}/eval_results_ch.json
  done
done
```

#### Step 4: Collect Results and Compute Statistics

```bash
python scripts/collect_results_with_stats.py \
  --base_dir ./outputs/multi_seed_experiments \
  --task sst2 \
  --methods full_ft lora_diffusion weight_lora adapters bitfit \
  --seeds "42-51" \
  --output collected_results_with_stats.json \
  --baseline full_ft
```

This generates:
- `collected_results_with_stats.json` - Full statistics in JSON format
- `collected_results_with_stats.summary.txt` - Human-readable summary

#### Step 5: Generate Paper Tables

```bash
python scripts/generate_paper_tables.py \
  --results collected_results_with_stats.json \
  --output paper_tables_with_stats.tex
```

This generates LaTeX tables with:
- Mean ± std for all metrics
- Significance markers (*, **, ***)
- Detailed statistics table with p-values and Cohen's d

## Understanding the Output

### Summary Report

The `collected_results_with_stats.summary.txt` contains:

```
VALIDATION ACCURACY:
--------------------------------------------------------------------------------
full_ft             :  51.15 ±  0.73%  95% CI: [ 50.69,  51.61]  n=10
lora_diffusion      :  48.97 ±  0.82%  95% CI: [ 48.40,  49.54]  n=10
                      vs baseline: p=0.0234 (**), Cohen's d=0.682
weight_lora         :  48.62 ±  1.05%  95% CI: [ 47.87,  49.37]  n=10
                      vs baseline: p=0.0081 (**), Cohen's d=0.823
...

STATISTICAL ASSUMPTIONS:
--------------------------------------------------------------------------------
Homogeneity of variance: Yes (Levene p=0.3421)
full_ft             : Normal=Yes (Shapiro-Wilk p=0.8234)
lora_diffusion      : Normal=Yes (Shapiro-Wilk p=0.6712)
...
```

### Interpreting Results

**Significance markers:**
- `*` : p < 0.05 (significant at 5% level)
- `**` : p < 0.01 (highly significant)
- `***` : p < 0.001 (very highly significant)
- `ns` : not significant (p ≥ 0.05)

**Effect sizes (Cohen's d):**
- |d| < 0.2: negligible
- 0.2 ≤ |d| < 0.5: small
- 0.5 ≤ |d| < 0.8: medium
- |d| ≥ 0.8: large

**Confidence intervals:**
- If CIs overlap: methods are statistically similar
- If CIs don't overlap: methods are clearly different

## File Structure

```
outputs/multi_seed_experiments/
├── sst2_full_ft_seed42/
│   ├── training_summary.json          # Training metrics
│   ├── evaluation_history.json        # Eval during training
│   ├── eval_results.json              # Generation-based eval
│   └── eval_results_ch.json           # Classification-head eval
├── sst2_full_ft_seed43/
├── ... (50 total experiment directories)
└── collected_results_with_stats.json  # Aggregated statistics
```

## Statistical Methods

### Paired T-Test

We use paired t-tests because each seed represents a matched observation across methods (same data, same initialization scheme, only method differs).

**Null hypothesis**: H₀: μₐ = μᵦ (methods have equal mean performance)
**Alternative**: H₁: μₐ ≠ μᵦ (two-tailed test)

### Bonferroni Correction

When comparing LoRA-Diffusion against 4 baselines, we perform 4 tests. To control family-wise error rate at α=0.05:

α_corrected = 0.05 / 4 = 0.0125

A result is significant if p < 0.0125.

### Cohen's d

Effect size measures the magnitude of difference in standard deviation units:

d = (μₐ - μᵦ) / σ_pooled

Where σ_pooled = √((σₐ² + σᵦ²) / 2)

## Troubleshooting

### Missing Results

If some experiments fail, the collection script will warn you:

```
Warning: Missing output for sst2_adapters_seed45
```

You can re-run individual experiments:

```bash
python scripts/train.py \
  --config configs/base_config.yaml \
  --task sst2 \
  --method adapters \
  --seed 45 \
  --output_dir outputs/multi_seed_experiments/sst2_adapters_seed45
```

### Insufficient Seeds

If you have fewer than 10 seeds, statistical power decreases. Minimum recommended: 5 seeds.

To run with fewer seeds:

```bash
python scripts/run_experiments.py \
  --tasks sst2 \
  --methods lora_diffusion \
  --num_seeds 5 \
  --start_seed 42
```

### Non-Normal Distributions

If Shapiro-Wilk test indicates non-normality (p < 0.05), the Wilcoxon signed-rank test results (also computed) should be used instead of t-test.

## Paper Integration

After running the pipeline:

1. **Review statistics**: Check `collected_results_with_stats.summary.txt`
2. **Update tables**: Copy tables from `paper_tables_with_stats.tex` to `doc/Paper.tex`
3. **Verify values**: Ensure mean±std values match between summary and paper
4. **Interpret results**: Update the "Statistical Significance and Effect Sizes" subsection with actual p-values and effect sizes
5. **Compile paper**: `cd doc && pdflatex Paper.tex && bibtex Paper && pdflatex Paper.tex && pdflatex Paper.tex`

## Expected Runtime

With 4×A100 GPUs:
- Training: ~2-3 hours per experiment × 50 experiments = **100-150 hours** (4-6 days)
- Evaluation: ~5 minutes per checkpoint × 50 = **4 hours**
- Statistics: ~1 minute

**Total**: ~5-7 days for complete pipeline

To speed up:
- Run methods in parallel on different GPUs
- Use SLURM array jobs (see `slurm/sweep_array.slurm`)
- Reduce training steps for preliminary results

## Citation

When reporting these results, cite the statistical methods:

```latex
We performed paired t-tests with Bonferroni correction \citep{bonferroni1936teoria}
and computed Cohen's d effect sizes \citep{cohen1988statistical}.
```

## Contact

For questions about the statistical analysis pipeline, see:
- `src/utils/statistical_analysis.py` - Core statistical functions
- `scripts/collect_results_with_stats.py` - Results aggregation
- `scripts/generate_paper_tables.py` - Table generation
