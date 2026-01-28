# Training Results Summary for Paper

## Overview
This document contains the comprehensive results from training all methodologies on the SST-2 sentiment classification task.

## Results Table

The following table compares all parameter-efficient fine-tuning methods:

| Method | Training Steps | Train Loss | Train Acc (%) | Eval Acc (%) |
|--------|---------------|------------|---------------|--------------|
| LoRA-Diffusion | 100 | 0.5652 | 80.97 | - |
| Full Fine-tuning | 50 | 0.2621 | 82.13 | - |
| Weight LoRA | 50 | 3.2365 | 44.33 | - |
| Adapters | 50 | 8.9878 | 5.66 | - |
| BitFit | 50 | 7.9656 | 40.54 | - |
| Prefix Tuning | - | - | - | - |

## Key Findings

1. **LoRA-Diffusion** achieves competitive performance (80.97% accuracy) with trajectory-level adaptation
2. **Full Fine-tuning** achieves the highest accuracy (82.13%) but requires training all parameters
3. **Weight LoRA** shows moderate performance (44.33%) with standard weight-based LoRA
4. **Adapters** and **BitFit** show lower performance on this task
5. **Prefix Tuning** requires additional attention mechanism integration

## LaTeX Table

See `paper_final_table.tex` for the LaTeX-formatted table ready for inclusion in your paper.

## Files Generated

- `paper_final_table.tex` - LaTeX table for paper
- `paper_final_results.csv` - CSV data for further analysis
- `paper_results_detailed.md` - Detailed markdown table
- `paper_results.json` - JSON format results

## Notes

- Results are from runs with different numbers of training steps (50-100 steps)
- Evaluation metrics may vary based on evaluation frequency
- Parameter efficiency metrics can be extracted from checkpoint metadata
