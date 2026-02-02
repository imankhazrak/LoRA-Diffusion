# Pre-Experiment Verification Checklist

This checklist verifies all code and documentation modifications are complete before running experiments.

## Code Modifications ✅

### 1. Tokenizer Configuration
- [x] `configs/base_config.yaml` - Added `tokenizer.name` configuration
- [x] `scripts/train.py` - Loads tokenizer from config (line 277)
- [x] `scripts/evaluate.py` - Loads tokenizer from config (line 188)
- [x] `scripts/compose_tasks.py` - Loads tokenizer from config (line 108)
- [x] `examples/comprehensive_example.py` - Loads tokenizer from config (line 58)

### 2. SST-2 Evaluation Fixes
- [x] `src/training/trainer.py` - `evaluate()` method generates text and computes task-specific metrics (lines 288-329)
- [x] `src/evaluation/metrics.py` - Added `decode_classification_label()` function for robust label matching
- [x] `src/evaluation/metrics.py` - Updated `compute_accuracy()` to use task config for label decoding
- [x] `src/evaluation/metrics.py` - Updated `compute_metrics()` to pass task_config to classification metrics

### 3. Baseline Improvements
- [x] `src/models/baselines.py` - Fixed missing `import math` at top level
- [x] `configs/methods/weight_lora.yaml` - Extended target_modules to include FFN layers (`intermediate.dense`, `output.dense`)
- [x] `src/models/baselines.py` - Improved module matching to avoid duplicates

### 4. Parameter Counting
- [x] `scripts/count_parameters.py` - Created unified parameter counting script
- [x] Script counts all methods: full_ft, lora_diffusion, weight_lora, adapters, bitfit, prefix_tuning
- [x] Script estimates storage sizes
- [x] Script outputs JSON and summary table

### 5. Trainer Improvements
- [x] `src/training/trainer.py` - Fixed metrics access to handle missing config gracefully (line 235-238)
- [x] `src/training/trainer.py` - Evaluation now generates text for task-specific metrics

## Documentation Modifications ✅

### 1. Paper Methodology (Section 3.2)
- [x] Clarified `A_i(c)` implementation via FiLM-style conditioning
- [x] Added equation showing `A_i(c)v = W_A^{(i)}(γ_i(c) ⊙ v) + β_i(c)`
- [x] Explained phase-shared adapter design (three banks reused across timesteps)

### 2. Paper Experimental Setup (Section 4.1)
- [x] Added explicit SST-2 formulation description
- [x] Clarified instruction template format
- [x] Explained text generation and label matching procedure
- [x] Updated to say "validation accuracy" instead of "accuracy"
- [x] Updated weight LoRA description to include MLP layers

### 3. Paper Results Tables
- [x] Table 3 (`tab:per_task_results`) - Updated to show "Val acc." instead of "Train acc." and "Eval acc."
- [x] Added note that all accuracies are validation-set accuracies
- [x] Removed empty "Eval acc." column

### 4. Paper Theory Section (Section 3.6)
- [x] Clarified "effective rank" vs classical matrix rank
- [x] Explained entropy-based effective rank definition
- [x] Added explanation of nuclear-norm and orthogonality regularization roles

### 5. Paper Related Work (Section 2.2)
- [x] Added paragraph comparing to T-LoRA, FouRA, SeLoRA, GeLoRA, EST-LoRA
- [x] Explained differences (trajectory space vs weight/frequency space)
- [x] Added placeholder citations in `doc/reference.bib`

### 6. Paper Conclusion (Section 6)
- [x] Added explicit limitations section
- [x] Acknowledged single task/single model size evaluation
- [x] Noted need for multi-task quantitative results
- [x] Mentioned heuristic rank schedule

## Configuration Files ✅

### Task Configs
- [x] `configs/tasks/sst2.yaml` - Has `metrics.primary: "accuracy"`
- [x] `configs/tasks/agnews.yaml` - Has `metrics.primary: "accuracy"`
- [x] `configs/tasks/squad.yaml` - Has `metrics.primary: "f1"`
- [x] `configs/tasks/xsum.yaml` - Has `metrics.primary: "rouge_l"`

### Method Configs
- [x] `configs/methods/weight_lora.yaml` - Updated with FFN target modules
- [x] All method configs exist and are properly structured

## Remaining Tasks (Before Experiments)

### Critical
1. **Run parameter counting script**:
   ```bash
   python scripts/count_parameters.py \
       --config configs/base_config.yaml \
       --task sst2 \
       --output parameter_counts.json
   ```

2. **Update paper tables** with consistent numbers from parameter counting script:
   - Table 3 (`tab:main_results`) - Update parameter percentages
   - Table 4 (`tab:efficiency`) - Update storage sizes
   - Any other efficiency tables

3. **Fill in bibliography citations** in `doc/reference.bib`:
   - T-LoRA: Add actual authors and verify arXiv number
   - FouRA: Verify arXiv:2406.08798
   - SeLoRA: Verify arXiv:2408.07196
   - GeLoRA: Verify arXiv:2412.09250
   - EST-LoRA: Verify arXiv:2508.02165

### Before Re-running Experiments
4. **Verify evaluation works correctly**:
   - Run a small test: `python scripts/train.py --task sst2 --method lora_diffusion --max_steps 10 --subset_size 100`
   - Check that evaluation generates text and computes accuracy
   - Verify label decoding works correctly

5. **Check baseline implementations**:
   - Verify weight LoRA applies to both attention and FFN
   - Verify adapters are inserted correctly
   - Verify BitFit only trains biases

## Verification Commands

### Test tokenizer configuration:
```bash
python -c "
from src.utils import load_config
from pathlib import Path
config = load_config(Path('configs/base_config.yaml'), task_name='sst2')
print('Tokenizer:', config.get('tokenizer', {}).get('name', 'NOT FOUND'))
"
```

### Test parameter counting:
```bash
python scripts/count_parameters.py \
    --config configs/base_config.yaml \
    --task sst2 \
    --methods lora_diffusion weight_lora \
    --output test_counts.json
```

### Test evaluation:
```bash
# Small test run
python scripts/train.py \
    --task sst2 \
    --method lora_diffusion \
    --max_steps 20 \
    --subset_size 50 \
    --eval_frequency 10
```

## Files Modified Summary

### Code Files (9 files)
1. `src/training/trainer.py` - Evaluation improvements, metrics handling
2. `src/evaluation/metrics.py` - Label decoding, task config support
3. `src/models/baselines.py` - Math import fix, module matching
4. `scripts/train.py` - Tokenizer config
5. `scripts/evaluate.py` - Tokenizer config
6. `scripts/compose_tasks.py` - Tokenizer config
7. `scripts/count_parameters.py` - NEW: Parameter counting
8. `examples/comprehensive_example.py` - Tokenizer config
9. `configs/base_config.yaml` - Tokenizer config
10. `configs/methods/weight_lora.yaml` - FFN target modules

### Documentation Files (2 files)
1. `doc/Paper.tex` - Multiple sections updated
2. `doc/reference.bib` - Added placeholder citations

### Summary Files (2 files)
1. `REVISION_SUMMARY.md` - Detailed revision summary
2. `PRE_EXPERIMENT_CHECKLIST.md` - This file

## Status: ✅ READY FOR EXPERIMENTS

All code modifications are complete. Documentation is updated. Remaining tasks are:
1. Run parameter counting script
2. Update paper tables with consistent numbers
3. Fill in bibliography citations
4. Run experiments with fixed evaluation
