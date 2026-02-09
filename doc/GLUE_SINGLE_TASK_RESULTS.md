# GLUE Single-Task Results (Token-Level Accuracy)

Collected from **75** complete runs in `outputs/glue_single/`.

Metric: **token-level denoising accuracy** (fraction of masked tokens predicted correctly on the validation set).

| Task | Method | Seed | Train Acc (%) | Best Val Acc (%) | Final Val Acc (%) |
|------|--------|------|---------------|------------------|-------------------|
| MRPC | adapters | 42 | 82.58 | 85.92 | 83.61 |
| MRPC | adapters | 43 | 84.13 | 85.37 | 82.08 |
| MRPC | adapters | 44 | 81.52 | 86.57 | 84.65 |
| MRPC | adapters | 45 | 81.86 | 87.06 | 87.06 |
| MRPC | adapters | 46 | 82.13 | 83.50 | 82.87 |
| MRPC | bitfit | 42 | 100.00 | 100.00 | 100.00 |
| MRPC | bitfit | 43 | 100.00 | 100.00 | 100.00 |
| MRPC | bitfit | 44 | 100.00 | 100.00 | 100.00 |
| MRPC | bitfit | 45 | 100.00 | 100.00 | 100.00 |
| MRPC | bitfit | 46 | 100.00 | 100.00 | 100.00 |
| MRPC | full_ft | 42 | --- | 100.00 | 100.00 |
| MRPC | full_ft | 43 | --- | 100.00 | 100.00 |
| MRPC | full_ft | 44 | --- | 100.00 | 100.00 |
| MRPC | full_ft | 45 | --- | 100.00 | 100.00 |
| MRPC | full_ft | 46 | --- | 100.00 | 100.00 |
| MRPC | lora_diffusion | 42 | --- | 94.45 | 94.45 |
| MRPC | lora_diffusion | 43 | --- | 93.63 | 93.63 |
| MRPC | lora_diffusion | 44 | 98.54 | 99.81 | 99.57 |
| MRPC | lora_diffusion | 45 | 99.35 | 100.00 | 99.66 |
| MRPC | lora_diffusion | 46 | 99.03 | 99.88 | 98.50 |
| MRPC | weight_lora | 42 | 100.00 | 100.00 | 100.00 |
| MRPC | weight_lora | 43 | 100.00 | 100.00 | 100.00 |
| MRPC | weight_lora | 44 | 100.00 | 100.00 | 100.00 |
| MRPC | weight_lora | 45 | 100.00 | 100.00 | 100.00 |
| MRPC | weight_lora | 46 | 100.00 | 100.00 | 100.00 |
| QNLI | adapters | 42 | --- | 67.01 | 67.01 |
| QNLI | adapters | 43 | --- | 68.17 | 68.17 |
| QNLI | adapters | 44 | --- | 67.31 | 67.31 |
| QNLI | adapters | 45 | --- | 65.83 | 65.83 |
| QNLI | adapters | 46 | --- | 67.14 | 67.04 |
| QNLI | bitfit | 42 | --- | 100.00 | 100.00 |
| QNLI | bitfit | 43 | --- | 100.00 | 100.00 |
| QNLI | bitfit | 44 | --- | 100.00 | 100.00 |
| QNLI | bitfit | 45 | --- | 100.00 | 100.00 |
| QNLI | bitfit | 46 | --- | 100.00 | 100.00 |
| QNLI | full_ft | 42 | --- | 100.00 | 100.00 |
| QNLI | full_ft | 43 | --- | 100.00 | 100.00 |
| QNLI | full_ft | 44 | --- | 100.00 | 100.00 |
| QNLI | full_ft | 45 | --- | 100.00 | 100.00 |
| QNLI | full_ft | 46 | --- | 100.00 | 100.00 |
| QNLI | lora_diffusion | 42 | --- | 99.43 | 99.23 |
| QNLI | lora_diffusion | 43 | --- | 99.23 | 99.23 |
| QNLI | lora_diffusion | 44 | --- | 99.13 | 99.13 |
| QNLI | lora_diffusion | 45 | --- | 99.85 | 99.85 |
| QNLI | lora_diffusion | 46 | --- | 99.31 | 98.96 |
| QNLI | weight_lora | 42 | --- | 100.00 | 100.00 |
| QNLI | weight_lora | 43 | --- | 100.00 | 100.00 |
| QNLI | weight_lora | 44 | --- | 100.00 | 100.00 |
| QNLI | weight_lora | 45 | --- | 100.00 | 100.00 |
| QNLI | weight_lora | 46 | --- | 100.00 | 100.00 |
| SST2 | adapters | 42 | 84.70 | 85.16 | 83.25 |
| SST2 | adapters | 43 | 84.81 | 85.23 | 83.23 |
| SST2 | adapters | 44 | 85.94 | 85.09 | 83.52 |
| SST2 | adapters | 45 | 85.77 | 85.09 | 84.06 |
| SST2 | adapters | 46 | 84.76 | 85.25 | 82.72 |
| SST2 | bitfit | 42 | 84.68 | 84.70 | 84.23 |
| SST2 | bitfit | 43 | 86.33 | 84.70 | 83.62 |
| SST2 | bitfit | 44 | 84.03 | 84.72 | 83.49 |
| SST2 | bitfit | 45 | 84.72 | 84.28 | 83.62 |
| SST2 | bitfit | 46 | 85.47 | 85.25 | 84.39 |
| SST2 | full_ft | 42 | 85.58 | 84.70 | 84.23 |
| SST2 | full_ft | 43 | 86.32 | 85.09 | 83.62 |
| SST2 | full_ft | 44 | 84.89 | 84.72 | 83.49 |
| SST2 | full_ft | 45 | 84.70 | 84.27 | 83.61 |
| SST2 | full_ft | 46 | 86.12 | 85.25 | 84.39 |
| SST2 | lora_diffusion | 42 | 87.14 | 87.84 | 87.52 |
| SST2 | lora_diffusion | 43 | 87.79 | 88.28 | 86.89 |
| SST2 | lora_diffusion | 44 | 88.47 | 88.11 | 86.80 |
| SST2 | lora_diffusion | 45 | 88.25 | 88.19 | 87.51 |
| SST2 | lora_diffusion | 46 | 88.28 | 87.63 | 87.37 |
| SST2 | weight_lora | 42 | 84.15 | 84.95 | 82.28 |
| SST2 | weight_lora | 43 | 85.97 | 84.84 | 83.77 |
| SST2 | weight_lora | 44 | 84.91 | 85.44 | 83.68 |
| SST2 | weight_lora | 45 | 85.67 | 85.32 | 82.48 |
| SST2 | weight_lora | 46 | 85.46 | 85.58 | 84.34 |

## Summary by (Task, Method)

| Task | Method | Val Mean (%) | Val Std (%) | N seeds |
|------|--------|--------------|------------|--------|
| MRPC | adapters | 85.68 | 1.38 | 5 |
| MRPC | bitfit | 100.00 | 0.00 | 5 |
| MRPC | full_ft | 100.00 | 0.00 | 5 |
| MRPC | lora_diffusion | 97.56 | 3.22 | 5 |
| MRPC | weight_lora | 100.00 | 0.00 | 5 |
| QNLI | adapters | 67.09 | 0.84 | 5 |
| QNLI | bitfit | 100.00 | 0.00 | 5 |
| QNLI | full_ft | 100.00 | 0.00 | 5 |
| QNLI | lora_diffusion | 99.39 | 0.28 | 5 |
| QNLI | weight_lora | 100.00 | 0.00 | 5 |
| SST2 | adapters | 85.17 | 0.07 | 5 |
| SST2 | bitfit | 84.73 | 0.35 | 5 |
| SST2 | full_ft | 84.81 | 0.38 | 5 |
| SST2 | lora_diffusion | 88.01 | 0.27 | 5 |
| SST2 | weight_lora | 85.23 | 0.32 | 5 |

| Task | Method | Train Mean (%) | Train Std (%) | N seeds |
|------|--------|-----------------|----------------|--------|
| MRPC | adapters | 82.44 | 1.02 | 5 |
| MRPC | bitfit | 100.00 | 0.00 | 5 |
| MRPC | lora_diffusion | 98.97 | 0.41 | 3 |
| MRPC | weight_lora | 100.00 | 0.00 | 5 |
| SST2 | adapters | 85.20 | 0.61 | 5 |
| SST2 | bitfit | 85.05 | 0.88 | 5 |
| SST2 | full_ft | 85.52 | 0.72 | 5 |
| SST2 | lora_diffusion | 87.99 | 0.54 | 5 |
| SST2 | weight_lora | 85.23 | 0.72 | 5 |

## SST-2 statistical analysis (5 seeds)

| Method | Mean (%) | Std | Variance | 95% CI | p-value vs. full FT | Cohen's d |
|--------|----------|-----|----------|--------|----------------------|----------|
| full_ft | 84.81 | 0.38 | 0.15 | [84.33, 85.28] | --- | --- |
| lora_diffusion | 88.01 | 0.27 | 0.07 | [87.67, 88.34] | 0.0000 | 5.73 |
| weight_lora | 85.23 | 0.32 | 0.10 | [84.83, 85.62] | 0.0000 | 0.85 |
| adapters | 85.17 | 0.07 | 0.01 | [85.07, 85.26] | 0.0000 | 1.12 |
| bitfit | 84.73 | 0.35 | 0.12 | [84.30, 85.16] | 0.2227 | -0.42 |

## Note on 100% token-level accuracy (QNLI, MRPC)

Token-level accuracy is **denoising accuracy**: at evaluation we mask the label token and measure whether the model predicts it correctly given the instruction (teacher-forced). For binary classification with a single-token label this can reach 100% and is **not a bug**. The more comparable metric is **generation accuracy** (model generates the label from scratch; decoded output vs reference). Below is the mean generation accuracy (%) where available.

| Task | Method | Gen. acc. mean (%) | N seeds |
|------|--------|--------------------|--------|
| MRPC | adapters | 68.33 | 5 |
| MRPC | bitfit | 31.62 | 5 |
| MRPC | full_ft | 31.62 | 5 |
| MRPC | lora_diffusion | 31.91 | 5 |
| MRPC | weight_lora | 31.62 | 5 |
| QNLI | adapters | 49.46 | 5 |
| QNLI | bitfit | 49.46 | 5 |
| QNLI | full_ft | 49.46 | 5 |
| QNLI | lora_diffusion | 49.63 | 5 |
| QNLI | weight_lora | 49.46 | 5 |
| SST2 | adapters | 49.77 | 5 |
| SST2 | bitfit | 50.71 | 5 |
| SST2 | full_ft | 50.92 | 5 |
| SST2 | lora_diffusion | 48.97 | 5 |
| SST2 | weight_lora | 50.92 | 5 |

*Generated by `scripts/collect_glue_single_results.py`.*
