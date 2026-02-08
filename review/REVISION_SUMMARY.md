# LoRA-Diffusion Revision Summary

This document summarizes the improvements made to address reviewer concerns and improve the paper and codebase.

## Completed Improvements

### 1. Paper Methodology Alignment ✅
- **Updated `doc/Paper.tex` Section 3.2**: Clarified that `A_i(c)` is implemented via FiLM-style conditioning (instruction-dependent scale and shift), not a static matrix
- **Added clarification**: Explained that adapters are phase-shared (three banks for early/mid/late phases reused across timesteps), keeping parameter count independent of T
- **Impact**: Paper now accurately reflects the implementation in `src/models/lora_modules.py`

### 2. SST-2 Evaluation Fixes ✅
- **Updated `src/training/trainer.py`**: Modified `evaluate()` method to generate text and compute task-specific metrics (classification accuracy) during training evaluation
- **Added robust label decoding**: Created `decode_classification_label()` in `src/evaluation/metrics.py` to handle variations in generated text (e.g., "the sentiment is positive" → "positive")
- **Updated `src/evaluation/metrics.py`**: Enhanced `compute_accuracy()` to use task config for robust label matching
- **Updated paper Section 4.1**: Added explicit description of SST-2 formulation (instruction-following format, text generation, label matching)
- **Updated paper tables**: Clarified that all reported accuracies are validation-set accuracies computed via text generation
- **Impact**: Evaluation now consistently computes task-specific metrics, and paper clearly explains how SST-2 is evaluated with diffusion models

### 3. Baseline Improvements ✅
- **Fixed `src/models/baselines.py`**: Added missing `import math` at top level (was only imported inside function)
- **Updated `configs/methods/weight_lora.yaml`**: Extended target modules to include FFN layers (`intermediate.dense` and `output.dense`) in addition to attention layers
- **Updated paper Section 4.1**: Clarified that weight LoRA is applied to both attention (Q/K/V/O) and MLP layers
- **Impact**: Baselines are now more fairly implemented and documented

### 4. Parameter Accounting ✅
- **Created `scripts/count_parameters.py`**: Unified script to count trainable parameters and storage sizes for all PEFT methods
- **Script features**:
  - Counts base model parameters
  - Counts trainable parameters for each method (full FT, LoRA-Diffusion, weight LoRA, adapters, BitFit, prefix)
  - Estimates storage size in MB
  - Provides breakdown for LoRA-Diffusion
  - Outputs JSON and summary table
- **Impact**: Provides consistent parameter counting that can be used to regenerate efficiency tables

### 5. Theory and Related Work ✅
- **Updated `doc/Paper.tex` Section 3.6**: 
  - Clarified "effective rank" vs classical matrix rank
  - Explained that conditioning through `A(c)` changes the notion of rank
  - Added explanation of nuclear-norm and orthogonality regularization roles
- **Updated `doc/Paper.tex` Section 2.2**: Added paragraph comparing LoRA-Diffusion to T-LoRA, FouRA, SeLoRA, GeLoRA, EST-LoRA
- **Updated `doc/reference.bib`**: Added placeholder citations for new related work (user needs to fill in exact details)
- **Impact**: Theory is more precise, and paper properly positions against recent timestep-aware PEFT methods

### 6. Reproducibility Improvements ✅
- **Updated `configs/base_config.yaml`**: Added `tokenizer.name` configuration option
- **Updated `scripts/train.py`**: Loads tokenizer from config instead of hardcoding
- **Updated `scripts/evaluate.py`**: Loads tokenizer from config instead of hardcoding
- **Impact**: Tokenizer choice is now configurable and consistent across scripts

### 7. Results and Limitations ✅
- **Updated `doc/Paper.tex` Conclusion**: Added explicit limitations section covering:
  - Single task/single model size evaluation
  - Need for multi-task quantitative results
  - Heuristic rank schedule (could use principled allocation)
  - Need for nuclear-norm ablation analysis
  - Incomplete prefix tuning integration
- **Impact**: Paper is more honest about scope and limitations

## Remaining Tasks

### High Priority
1. **Run parameter counting script**: Execute `scripts/count_parameters.py` to generate consistent parameter/storage numbers and update efficiency tables in `doc/Paper.tex`
2. **Re-run experiments**: With fixed evaluation, re-run SST-2 experiments to get credible baseline results (especially if previous baselines were too low due to bugs)
3. **Update paper tables**: Replace placeholder/conflicting numbers in Tables 3, 4, and efficiency tables with values from parameter counting script

### Medium Priority
4. **Add multi-task experiment documentation**: Document how to run multi-task composition experiments (structure exists in code, needs README/paper documentation)
5. **Add ablation scripts**: Create `scripts/run_ablations.py` or extend `run_experiments.py` with ablation flags (rank schedules, orthogonality on/off, nuclear norm on/off)
6. **Fill in bibliography**: Update placeholder citations in `doc/reference.bib` with actual author names and verify arXiv numbers for T-LoRA, FouRA, SeLoRA, GeLoRA, EST-LoRA

### Low Priority
7. **Add integration tests**: Extend `tests/` to cover weight LoRA/adapters on diffusion model, SST-2 label decoding, multi-task composition smoke test
8. **Update README/QUICKSTART**: Reflect new tokenizer configuration and evaluation improvements

## Files Modified

### Code
- `src/training/trainer.py` - Added text generation and task metrics to evaluation
- `src/evaluation/metrics.py` - Added robust label decoding for classification
- `src/models/baselines.py` - Fixed math import, improved module matching
- `scripts/train.py` - Made tokenizer configurable
- `scripts/evaluate.py` - Made tokenizer configurable
- `scripts/count_parameters.py` - NEW: Unified parameter counting script
- `configs/base_config.yaml` - Added tokenizer configuration
- `configs/methods/weight_lora.yaml` - Extended to include MLP layers

### Paper
- `doc/Paper.tex` - Multiple sections updated (methodology, experimental setup, theory, related work, conclusion)
- `doc/reference.bib` - Added placeholder citations for related work

## Next Steps for User

1. **Run parameter counting**:
   ```bash
   python scripts/count_parameters.py --config configs/base_config.yaml --task sst2 --output parameter_counts.json
   ```

2. **Update paper tables** with consistent numbers from the counting script

3. **Re-run SST-2 experiments** with fixed evaluation to get credible baseline results:
   ```bash
   python scripts/train.py --task sst2 --method full_ft --max_steps 5000
   python scripts/train.py --task sst2 --method lora_diffusion --max_steps 5000
   python scripts/train.py --task sst2 --method weight_lora --max_steps 5000
   # ... etc for other baselines
   ```

4. **Fill in bibliography** with actual citation details for T-LoRA, FouRA, SeLoRA, GeLoRA, EST-LoRA

5. **Add multi-task experiments** if compute allows (structure is ready in code)

## Notes

- The codebase now has consistent evaluation that generates text and computes task-specific metrics
- Parameter counting is unified and can generate consistent efficiency tables
- Paper methodology accurately reflects implementation
- Baselines are more fairly implemented (weight LoRA includes MLP layers)
- Theory and related work sections are more precise and complete
- Limitations are explicitly acknowledged

The main remaining work is experimental: re-running with fixed evaluation and updating tables with consistent parameter counts.
