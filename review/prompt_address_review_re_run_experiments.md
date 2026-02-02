# Claude Prompt: Address Reviewer Comments and Re-run Experiments

## Role
You are a senior machine learning researcher with deep expertise in diffusion models, parameter-efficient fine-tuning (LoRA, adapters, prefix tuning), and top-tier ML conference standards (NeurIPS / ICML / ICLR). You are also an experienced research engineer capable of rewriting, debugging, and re-running large-scale training and evaluation pipelines.

## Task
I am revising a research paper titled **“LoRA-Diffusion: Parameter-Efficient Fine-Tuning via Low-Rank Trajectory Decomposition.”** Below is a full reviewer report from a NeurIPS-style review.

Your task is to **address *all* reviewer comments thoroughly**, including conceptual clarification, methodological fixes, experimental redesign, baseline correction, parameter and storage accounting, multi-task validation, related-work positioning, and **rewriting and re-running code where necessary**.

Treat this as a **full paper revision for resubmission**, not a rebuttal.

---

## Inputs You Have

- The full reviewer report (provided below)
- The original method idea: trajectory-level low-rank perturbations for diffusion language models
- A diffusion LM backbone (SEDD-style masked diffusion)
- An existing LoRA-Diffusion codebase (assume it may contain bugs or flawed evaluation)

---

## Required Tasks

### 1. Diagnose Reviewer Concerns
For each reviewer criticism:
- Classify it as **fatal**, **fixable**, or **positioning-related**
- Explain **why the reviewer is correct** (or partially correct)
- State **exactly what must change** in the paper or experiments to resolve it

---

### 2. Redesign the Experimental Protocol
You must:
- Correctly formulate **SST-2 evaluation with a diffusion language model**
- Clearly distinguish **training, validation, and test accuracy**
- Identify and fix any **evaluation or decoding bugs** that could explain near-random baselines
- Propose a **credible training schedule** (training steps, batch size, number of seeds)

Explicitly specify:
- conditioning scheme
- label decoding method
- loss function
- evaluation metrics

---

### 3. Re-implement Baselines Fairly
Rewrite or fix code (pseudocode or real PyTorch where helpful) for:
- Weight-level LoRA (applied to Q/K/V/O **and MLP layers**, with tuned ranks)
- Adapter layers compatible with diffusion models
- BitFit
- Prefix tuning (fully implemented, not partial)

Ensure all baselines:
- are diffusion-aware
- are reasonably tuned
- achieve **credible, non-degenerate performance**

---

### 4. Reconcile Parameter and Storage Accounting
Produce:
- A **single, consistent accounting** of trainable parameters
- A clear explanation of:
  - per-step vs shared modules
  - per-task storage
  - total trainable parameter percentage

Eliminate all contradictions (e.g., 0.7% vs 2.9%, 9M vs 39M).

---

### 5. Strengthen or Downgrade Theory Responsibly
You must:
- Clarify what **low-rank** means in trajectory (representation) space
- Explain how conditional maps (e.g., A(c)) affect rank interpretation
- Either justify the **nuclear-norm regularization** or remove it
- Propose ablations isolating the effect of orthogonality regularization

Avoid overstated guarantees; keep the theory honest and well-scoped.

---

### 6. Add Multi-Task and Composition Experiments
Design experiments on **at least three tasks**, for example:
- SST-2 (classification)
- AGNews (classification)
- XSum or SQuAD (generation)

Include:
- quantitative multi-task results
- zero-shot composition tests
- interference analysis
- qualitative trajectory or embedding visualizations

---

### 7. Position Against Related Work
Explicitly compare or contrast with:
- T-LoRA
- FouRA
- SeLoRA
- GeLoRA
- EST-LoRA

Explain:
- what is genuinely new
- what is similar
- why trajectory-space adaptation is distinct from weight- or frequency-based PEFT

---

### 8. Rewrite Code if Necessary
If the existing code structure is flawed:
- rewrite the training loop
- rewrite the evaluation loop
- propose correct inference code

Provide **clear runnable pseudocode or PyTorch-style code** for:
- training
- evaluation
- multi-task composition

---

## Output Format
Respond in **clearly labeled sections**:

1. Reviewer Concern → Diagnosis
2. Corrected Experimental Design
3. Baseline Re-implementation
4. Parameter and Storage Accounting
5. Theory Clarification and Ablations
6. Multi-Task and Composition Experiments
7. Related-Work Positioning
8. Revised Code (or Pseudocode)
9. Final Checklist for Resubmission

Be explicit, technical, and honest. Assume the response will be used to **re-run real experiments on GPUs**.

---

## Reviewer Report
Summary The paper proposes LoRA-Diffusion, a parameter-efficient fine-tuning method tailored to diffusion-based language models. Rather than applying low-rank updates to model weights (as in standard LoRA), the method adds low-rank perturbations directly to the denoising trajectory at each timestep, with a stepadaptive rank schedule and an optional router for compositional multi-task inference. On SST-2 with a 1.3B-parameter diffusion language model, the authors report performance close to full fine-tuning while training a small fraction of parameters, and claim advantages over weight LoRA, adapters, and BitFit. Strengths Technical novelty and innovationExperimental rigor and validation Clarity of presentation Significance of contributions Weaknesses Technical limitations or concerns The trajectory-level viewpoint—learning low-rank perturbations in representation space across diffusion timesteps—is a conceptually interesting shift from weight-level PEFT and is aligned with the iterative nature of diffusion models. • The step-adaptive rank allocation across diffusion phases is intuitively motivated (higher capacity for earlier, more global steps; lower capacity for late, local refinement) and resonates with observations from related diffusion PEFT work in other modalities. • The formulation allows compositionality at inference (router-weighted superposition of task-specific trajectory modules), which is promising for multi-task use cases. • The work includes ablations (rank schedules, number of modules) and efficiency metrics (parameter counts, storage), which are the right types of analyses for PEFT. • The high-level problem framing and the core methodology are presented in a reasonably clear manner; the idea of decomposing the trajectory into a frozen base path plus a low-rank perturbation is easy to grasp. • If validated, a PEFT method specifically adapted to diffusion-based language models would fill a gap, since most PEFT literature has focused on autoregressive LMs or image diffusion models. • The proposed “low-rank perturbation” module is functionally close to adding small bottleneck residual MLPs at each step; the specific advantages over standard adapters inserted in diffusion stacks (beyond the step-aware schedule) are not theoretically or empirically disentangled. •Experimental gaps or methodological issues Clarity or presentation issues The “theoretical justification” via the information bottleneck is very highlevel; no formal analysis links the proposed objective or architecture to guaranteed low-rank trajectory structure or improved generalization. • The nuclear-norm regularization on already low-dimensional A/B factors (due to explicit rank r) is unclear and seems redundant; its practical effect is not analyzed. • Substantial internal inconsistencies exist in the reported numbers: LoRADiffusion is variously described as 0.7–1.0% parameters (~29M), 2.9% (39.6M), and 0.7% (9.1M) across tables. Storage numbers and parameter counts do not reconcile with the base model size. • The core results on SST-2 are not credible as presented. Baselines (adapters at 5.66% accuracy; BitFit 40.54%; LoRA 44.33%) perform far below random chance or known baselines for SST-2, suggesting integration or evaluation bugs. Full fine-tune accuracy of ~82% is also far below well-known results on SST-2 for large models. • The paper conflates training and evaluation metrics (e.g., Table 6 lists “Train acc.” as the headline accuracy while “Eval acc.” is missing), yet earlier tables claim evaluation accuracy. This severely undermines the claims. • Only one task (SST-2) is reported for quantitative results, despite multiple claims about multi-task composition; figures referencing SQuAD, XSum, AGNews are illustrative but not backed by experiments. • Prefix tuning is listed as a baseline but is “not fully implemented”; this weakens comparative claims. The LoRA baseline is limited (Q/K/V/O only) and does not explore MLPs or well-tuned settings common in the literature. • Training durations of 50–100 steps for a 1.3B model are implausibly short for stable convergence on text tasks; no variance or multi-seed runs are provided. • Some figures and tables are placeholders (“to be generated”), and the GitHub URL is a template, not a verifiable repository. Several tables contain artifacts (e.g., “checkmark symbol”). •Missing related work or comparisons Detailed Comments Technical soundness evaluation Experimental evaluation assessment The methodology is ambiguous about whether modules are shared across timesteps or learned per-step (and how this interacts with the step-adaptive ranks). • The paper does not situate itself against recent PEFT advances relevant to diffusion or timestep-aware capacity: T-LoRA (timestep-dependent rank masking and orthogonalization), FouRA (frequency-domain low-rank with input/timestep-dependent gating), SeLoRA (Fisher-informed adaptive rank), and diffusion adapter composition literature (e.g., EST-LoRA). It also omits recent rank-allocation theory (e.g., GeLoRA) that could better ground the step-adaptive schedule. This weakens the novelty and context. • The idea of applying a low-rank residual in representation space per diffusion step is sound in principle and can be seen as an adapter operating in the denoising loop. However, the paper does not rigorously argue why operating on x_t (trajectory space) is strictly preferable to operating on weights or intermediate features already present in the diffusion network. Without stronger analysis or controls, the method risks being an adapter-by-another-name with a timestep schedule. • The A(c) conditioning pushes the “low-rank” notion somewhat out of its classic linear-algebra framing (rank pertains to linear maps), since A depends on an encoder of c; the capacity of A(c) may exceed the nominal r×d parameterization. This should be clarified together with how the nuclear norm interacts with such conditionality. • The orthogonality regularizer across modules is reasonable, but its effect is not isolated in ablations beyond a brief mention; the paper should demonstrate its necessity and impact. • The reported SST-2 results contradict standard expectations and contain clear inconsistencies (parameter counts, storage, and accuracy). Baseline • S paperreview.ai By Stanford ML Group Upload Paper View Review Tech OverviewComparison with related work (using the summaries provided) accuracies far below chance strongly suggest an implementation error. The absence of proper evaluation metrics (test/validation accuracy) and the conflation with training accuracy further invalidate the conclusions. The experimental scope is too narrow to substantiate claims about multitask composition or generality. If the router and composition are central, at least two to three diverse NLP tasks (classification, QA, summarization) with quantitative metrics and qualitative analyses are necessary. • Efficiency: While the paper provides per-task storage estimates, it does not quantify runtime overhead at inference due to extra g_φ evaluations at every timestep, nor does it characterize memory footprint under realistic batch sizes and sequence lengths. Since methods like FouRA and SeLoRA also introduce overheads, a fair comparison should include latency/throughput profiling. • T-LoRA (2507.05964) directly addresses timestep-dependent adapter capacity in diffusion (albeit for images) with orthogonality to maintain effective rank. The proposed step-adaptive ranks are conceptually similar and should be compared and discussed; e.g., does LoRA-Diffusion’s trajectory injection outperform timestep-masked LoRA at equivalent budgets? • FouRA (2406.08798) introduces frequency-domain LoRA and adaptive rank gating across diffusion timesteps, with empirical gains and merging behavior. This is highly related both in spirit (timestep awareness, composition) and in goals (disentanglement, adapter merging). A discussion and empirical comparison are necessary. • SeLoRA (2408.07196) and GeLoRA (2412.09250) provide principled ways to allocate rank based on Fisher information and intrinsic dimension. The paper’s heuristic step schedule could be replaced or at least compared with such principled allocation. • EST-LoRA (2508.02165) studies training-free adapter fusion in diffusion via routing at inference. Since this paper also proposes composition (with a trainable router), comparisons on composition fidelity/conflicts would be valuable. • Solo Connection (2507.14353) reframes PEFT as representation-level cross-layer adaptation, which is philosophically close to the proposed •Discussion of broader impact and significance Questions for Authors 1. How exactly is SST-2 formulated and evaluated with a diffusion language model? Please specify the conditioning scheme, decoding to labels, splits, and whether reported accuracy is validation/test accuracy. Why are “Eval acc.” cells empty in Table 6? 2. Why do adapter, BitFit, and weight-LoRA baselines perform so poorly (below random chance)? Can you provide diagnostics, hyperparameter sweeps, and corrected results with credible baselines (including LoRA on MLP layers, tuned ranks, and prefix tuning once implemented)? 3. There are conflicting parameter and storage numbers (0.7% vs 2.9%; 29M vs 39.6M vs 9.1M). Please provide a single, consistent accounting of trainable parameters for the reported configuration, including whether modules are perstep, per-block, or shared across steps. 4. What is the runtime/latency overhead at inference for LoRA-Diffusion versus weight LoRA and adapters under equal accuracy? Please include throughput and memory usage across sequence lengths and batch sizes. 5. How does the method compare against timestep-aware LoRA variants (e.g., TLoRA) and frequency/gated PEFT (FouRA) on the same backbones/tasks? Can you replicate a subset of their setups or, at minimum, discuss key differences and expected trade-offs? 6. Can you provide quantitative multi-task and composition results (e.g., SST-2, AGNews, SQuAD, XSum) showing zero-shot composition performance and trajectory-level perturbation; acknowledging this connection would improve positioning. A PEFT method tailored to diffusion language models is potentially impactful, particularly if it enables efficient multi-task deployment. However, the current empirical foundation is too weak to support such claims. If validated with robust experiments, the trajectory-level perspective could inspire new PEFT designs that align more closely with iterative generative processes. •interference analysis? The current figures reference tasks beyond SST-2 but lack measured results. 7. What is the precise role and effect size of the nuclear norm and orthogonality regularizers given the explicit bottleneck r? Please add ablations isolating these terms. 8. Is A(c) a dynamic linear map with parameters dependent on c? If so, how is “rank” defined in this conditional setting, and does it undermine the low-rank interpretation? Overall Assessment The paper introduces a conceptually appealing idea—low-rank perturbations of the diffusion trajectory with step-adaptive capacity and compositionality—that could be impactful if validated. Unfortunately, the current experimental evidence is insufficient and internally inconsistent. The main quantitative results on SST-2 raise serious red flags (implausibly low baselines, conflation of train vs. eval accuracy, and contradictory parameter counts), and claims about multi-task composition are not substantiated by experiments. The theoretical section is heuristic and does not materially strengthen the case. Important related work on timestep-aware/adaptive-rank PEFT for diffusion (T-LoRA, FouRA, SeLoRA) and principled rank allocation (GeLoRA) is not adequately discussed or compared, which weakens the novelty positioning. Overall, while the high-level idea is promising and potentially significant, substantial revisions are needed: correct and expand the empirical evaluation across multiple tasks; provide fair, credible baselines; reconcile parameter/efficiency accounting; and better situate the method relative to recent PEFT advances. In its current form, I cannot recommend acceptance at NeurIPS.

---

### Important Constraints
- Do **not** hand-wave experimental fixes
- Do **not** assume baselines are weak by default
- If a failure is likely due to a bug, identify the bug and explain how to fix it
- Treat this as a serious NeurIPS-level revision

Begin now.

