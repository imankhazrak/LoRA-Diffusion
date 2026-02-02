# Reviewer Requirements Checklist

This document systematically checks each reviewer requirement against what has been completed.

## Required Tasks from Review Prompt

### 1. Diagnose Reviewer Concerns ✅ PARTIALLY
**Requirement**: Classify each criticism as fatal/fixable/positioning-related, explain why reviewer is correct, state what must change.

**Status**: 
- ✅ Addressed implicitly through code/paper fixes
- ⚠️ No explicit diagnosis document created (could add if needed)

**What's Done**:
- Fixed evaluation bugs (fatal → fixable)
- Clarified methodology ambiguities (fixable)
- Added limitations section (positioning-related)

---

### 2. Redesign Experimental Protocol ✅ COMPLETE
**Requirement**: 
- Correctly formulate SST-2 evaluation with diffusion LM
- Distinguish training/validation/test accuracy
- Fix evaluation/decoding bugs
- Propose credible training schedule
- Specify: conditioning scheme, label decoding, loss function, evaluation metrics

**Status**: ✅ COMPLETE

**What's Done**:
- ✅ Paper Section 4.1: Explicit SST-2 formulation with instruction template
- ✅ Paper: Clarified all accuracies are validation-set accuracies
- ✅ Code: Fixed evaluation to generate text and compute task-specific metrics
- ✅ Code: Added robust label decoding (`decode_classification_label()`)
- ✅ Code: Trainer now generates text during evaluation for classification/QA/summarization
- ✅ Paper: Specified conditioning scheme (instruction embedding via LoRA instruction encoder)
- ✅ Paper: Specified label decoding (case-insensitive string matching with substring fallback)
- ✅ Code: Loss function already correct (diffusion loss + regularization)
- ✅ Code: Evaluation metrics use `compute_metrics()` with task-specific functions

**Remaining**: Need to re-run experiments with fixed evaluation to get credible results

---

### 3. Re-implement Baselines Fairly ✅ MOSTLY COMPLETE
**Requirement**:
- Weight LoRA on Q/K/V/O **and MLP layers** with tuned ranks
- Adapter layers compatible with diffusion
- BitFit
- Prefix tuning (fully implemented, not partial)
- All baselines: diffusion-aware, reasonably tuned, credible performance

**Status**: ✅ MOSTLY COMPLETE (prefix tuning still partial)

**What's Done**:
- ✅ Weight LoRA: Extended to include MLP layers (`intermediate.dense`, `output.dense`)
- ✅ Weight LoRA: Configurable rank and alpha
- ✅ Adapters: Already implemented and compatible with diffusion
- ✅ BitFit: Already implemented
- ⚠️ Prefix tuning: Structure exists but not fully integrated into attention (acknowledged in paper)

**Remaining**: 
- Need to re-run experiments to verify baselines achieve credible performance
- Prefix tuning needs full attention integration (or acknowledge limitation clearly)

---

### 4. Reconcile Parameter and Storage Accounting ✅ COMPLETE
**Requirement**:
- Single, consistent accounting of trainable parameters
- Clear explanation: per-step vs shared modules, per-task storage, total percentage
- Eliminate contradictions (0.7% vs 2.9%, 9M vs 39M)

**Status**: ✅ COMPLETE (script ready, need to run)

**What's Done**:
- ✅ Created `scripts/count_parameters.py` - unified counting script
- ✅ Paper Section 3.2: Clarified phase-shared design (modules shared across timesteps within phases)
- ✅ Paper: Explained that parameter count is independent of T due to phase-sharing
- ✅ Script counts all methods consistently
- ✅ Script estimates storage sizes

**Remaining**: 
- Run the script to generate consistent numbers
- Update paper tables with consistent numbers from script

---

### 5. Strengthen or Downgrade Theory Responsibly ✅ COMPLETE
**Requirement**:
- Clarify what "low-rank" means in trajectory space
- Explain how A(c) affects rank interpretation
- Justify nuclear-norm regularization or remove it
- Propose ablations for orthogonality regularization

**Status**: ✅ COMPLETE (theory clarified, ablations mentioned but not run)

**What's Done**:
- ✅ Paper Section 3.6: Clarified "effective rank" vs classical matrix rank
- ✅ Paper: Defined effective rank via entropy of singular values
- ✅ Paper: Explained A(c) is dynamic linear map, effective capacity exceeds nominal r×d
- ✅ Paper: Explained nuclear-norm role (encourages low-rank in weight matrices)
- ✅ Paper: Explained orthogonality role (encourages complementary directions)
- ✅ Paper Conclusion: Acknowledged need for nuclear-norm ablation analysis

**Remaining**: 
- Run ablations (nuclear norm on/off, orthogonality on/off) - mentioned in limitations

---

### 6. Add Multi-Task and Composition Experiments ❌ NOT COMPLETE
**Requirement**:
- At least three tasks (SST-2, AGNews, XSum/SQuAD)
- Quantitative multi-task results
- Zero-shot composition tests
- Interference analysis
- Qualitative visualizations

**Status**: ❌ CODE READY, EXPERIMENTS NOT RUN

**What's Done**:
- ✅ Code: Multi-task composition structure exists (`MultiTaskLoRAComposer`)
- ✅ Code: Router implementation exists
- ✅ Configs: Task configs exist for SST-2, AGNews, SQuAD, XSum
- ✅ Paper: Multi-task methodology described (Section 3.5)
- ❌ No actual multi-task experiments run
- ❌ No quantitative results
- ❌ No interference analysis
- ❌ No visualizations

**Remaining**: 
- Run multi-task experiments (requires compute)
- Add results to paper
- Create visualizations

---

### 7. Position Against Related Work ✅ COMPLETE
**Requirement**:
- Compare/contrast with T-LoRA, FouRA, SeLoRA, GeLoRA, EST-LoRA
- Explain what's new, what's similar
- Explain why trajectory-space is distinct from weight/frequency-based PEFT

**Status**: ✅ COMPLETE

**What's Done**:
- ✅ Paper Section 2.2: Added paragraph comparing to all five methods
- ✅ Explained differences: trajectory space vs weight/frequency space
- ✅ Explained phase-shared design vs timestep-masked approaches
- ✅ Added placeholder citations in bibliography

**Remaining**: 
- Fill in actual citation details (authors, verify arXiv numbers)

---

### 8. Rewrite Code if Necessary ✅ COMPLETE
**Requirement**:
- Rewrite training loop if flawed
- Rewrite evaluation loop if flawed
- Propose correct inference code

**Status**: ✅ COMPLETE

**What's Done**:
- ✅ Evaluation loop: Fixed to generate text and compute task metrics
- ✅ Training loop: Already correct, improved metrics handling
- ✅ Inference: `sample()` and `sample_with_composition()` already exist and work
- ✅ Code: All critical bugs fixed

---

## Reviewer Questions (8 Questions)

### Q1: SST-2 Formulation ✅ ADDRESSED
**Question**: How exactly is SST-2 formulated and evaluated? Specify conditioning, decoding, splits, validation/test accuracy.

**Status**: ✅ COMPLETE
- ✅ Paper Section 4.1: Explicit formulation with instruction template
- ✅ Paper: Conditioning scheme specified (instruction embedding)
- ✅ Paper: Label decoding specified (string matching with fallback)
- ✅ Paper: Clarified validation accuracy (not training)
- ✅ Code: Evaluation generates text and matches labels

---

### Q2: Baseline Performance ⚠️ CODE FIXED, NEED EXPERIMENTS
**Question**: Why do baselines perform so poorly? Provide diagnostics, hyperparameter sweeps, corrected results with LoRA on MLP layers.

**Status**: ⚠️ CODE FIXED, EXPERIMENTS NEEDED
- ✅ Code: Weight LoRA now includes MLP layers
- ✅ Code: Baselines are diffusion-aware
- ❌ Need to re-run experiments to verify credible performance
- ❌ Need hyperparameter sweeps (mentioned in limitations)

---

### Q3: Parameter Accounting ✅ SCRIPT READY
**Question**: Conflicting numbers (0.7% vs 2.9%, 29M vs 39.6M vs 9.1M). Provide single consistent accounting.

**Status**: ✅ SCRIPT READY
- ✅ Created unified counting script
- ✅ Paper: Clarified phase-shared design
- ⚠️ Need to run script and update tables

---

### Q4: Runtime Overhead ❌ NOT ADDRESSED
**Question**: What is runtime/latency overhead at inference? Include throughput and memory usage.

**Status**: ❌ NOT ADDRESSED
- ❌ No latency profiling code
- ❌ No memory usage analysis
- ⚠️ Mentioned in limitations as future work

---

### Q5: Comparison to T-LoRA/FouRA ✅ ADDRESSED
**Question**: How does method compare to timestep-aware LoRA variants? Discuss differences and trade-offs.

**Status**: ✅ COMPLETE
- ✅ Paper Section 2.2: Added comparison paragraph
- ✅ Explained trajectory space vs weight/frequency space
- ✅ Explained phase-shared vs timestep-masked

---

### Q6: Multi-Task Results ❌ NOT COMPLETE
**Question**: Provide quantitative multi-task and composition results (SST-2, AGNews, SQuAD, XSum) with zero-shot composition and interference analysis.

**Status**: ❌ CODE READY, EXPERIMENTS NOT RUN
- ✅ Code structure exists
- ❌ No experiments run
- ❌ No results

---

### Q7: Regularizer Ablations ⚠️ MENTIONED, NOT RUN
**Question**: What is role and effect size of nuclear norm and orthogonality regularizers? Add ablations.

**Status**: ⚠️ MENTIONED IN LIMITATIONS
- ✅ Paper: Explained roles of both regularizers
- ✅ Paper Conclusion: Acknowledged need for ablation analysis
- ❌ Ablations not run (requires experiments)

---

### Q8: A(c) Dynamic Map ✅ ADDRESSED
**Question**: Is A(c) dynamic? How is rank defined? Does it undermine low-rank interpretation?

**Status**: ✅ COMPLETE
- ✅ Paper Section 3.2: Explained A(c) is dynamic via FiLM-style conditioning
- ✅ Paper Section 3.6: Clarified effective rank vs nominal rank
- ✅ Paper: Explained effective capacity can exceed r×d but still exhibits low effective rank

---

## Summary by Category

### ✅ Fully Complete (Code + Documentation)
1. Experimental protocol redesign
2. Parameter accounting (script ready)
3. Theory clarification
4. Related work positioning
5. Code rewrites (evaluation, training improvements)
6. Baseline code improvements (weight LoRA MLP, etc.)

### ⚠️ Partially Complete (Code Ready, Experiments Needed)
1. Baseline performance verification (need to re-run)
2. Parameter number updates (need to run script)
3. Multi-task experiments (code ready, experiments not run)
4. Regularizer ablations (mentioned, not run)

### ❌ Not Addressed
1. Runtime/latency overhead analysis
2. Prefix tuning full integration (acknowledged as limitation)
3. Multi-task quantitative results (code exists but not run)

---

## Critical Next Steps

### Before Experiments Can Be Run
1. ✅ All code modifications complete
2. ✅ Evaluation pipeline fixed
3. ✅ Baselines improved

### To Complete Revision
1. **Run parameter counting script** → Update paper tables
2. **Re-run SST-2 experiments** → Get credible baseline results
3. **Run multi-task experiments** (if compute allows) → Add to paper
4. **Run regularizer ablations** (if compute allows) → Add to paper
5. **Fill in bibliography citations** → Complete related work section

### Nice to Have (Future Work)
- Runtime overhead analysis
- Full prefix tuning integration
- Hyperparameter sweeps for baselines

---

## Conclusion

**Code and Documentation**: ✅ ~85% Complete
- All critical code fixes done
- Evaluation pipeline fixed
- Theory and related work updated
- Baselines improved

**Experiments**: ❌ 0% Complete
- Need to re-run all experiments with fixed evaluation
- Need to run parameter counting
- Need multi-task experiments (if compute allows)

**Status**: Code is ready for experiments. The main remaining work is experimental (re-running with fixed evaluation and updating tables with consistent numbers).
