You are my ML engineer + research assistant. I need you to modify this diffusion-LM fine-tuning repo to run a clean, reproducible experimental suite for LoRA-Diffusion and baselines, addressing reviewer issues (parameter fairness, metric clarity, table consistency). You MUST implement the code changes and create runnable scripts/jobs. Do not hand-wave; produce concrete diffs, new scripts, and commands.

GOAL
Run solid experiments on three GLUE tasks: SST-2, QNLI, MRPC.
We will run:
(A) single-task training jobs (one task at a time)
(B) multi-task training jobs (joint training on all three tasks with composition/router if supported)

RIGOR REQUIREMENTS
- Exactly 5 runs per setting with 5 fixed seeds: [42, 43, 44, 45, 46].
- Save per-run outputs + a final aggregated CSV/JSON with mean/std and 95% CI.
- Make results fully reproducible: log git commit hash, config, seed, hardware info, and exact command.

METRICS (IMPORTANT)
1) Token-level denoising accuracy (primary training-aligned metric) for train/val/test.
2) Standard GLUE task metrics WITHOUT adding a [CLS] linear head.
   - Implement a head-free evaluation:
     - decode the final x0 (or model prediction) into text,
     - map prediction to labels using a fixed verbalizer/decision rule.
   - For SST-2: label verbalizer {“positive”, “negative”} or {“yes”, “no”} (pick one and keep constant).
   - For QNLI: verbalizer {“entailment”, “not_entailment”}.
   - For MRPC: verbalizer {“paraphrase”, “not_paraphrase”} (or {“equivalent”, “not_equivalent”}). Keep constant.
   - Report Accuracy for SST-2/QNLI; for MRPC report Accuracy + F1 (standard GLUE), derived from the head-free rule.
3) Efficiency metrics (measured consistently):
   - trainable parameter count
   - checkpoint size on disk (MB)
   - training wall-clock time (total and per step)
   - inference latency (ms/example) under a fixed batch size and fixed diffusion steps T

FAIRNESS / PARAMETER ACCOUNTING (KEY REVIEWER ISSUE)
The reviewer complained LoRA-Diffusion “28.7%” is inflated by the instruction encoder.
Fix this by standardizing conditioning across ALL methods.

Implement TWO accounting views and report both in the final tables:
- View 1 (PEFT-only): counts ONLY the adaptation modules (e.g., trajectory adapters, weight LoRA matrices, adapter blocks, BitFit biases). The instruction encoder is excluded for all methods.
- View 2 (Total trainable): counts PEFT modules + instruction encoder IF the instruction encoder is trained.

To make this fair, implement/configure these two modes:
MODE-FROZEN-ENC: instruction encoder is frozen for all methods (preferred baseline fairness).
MODE-TRAIN-ENC: instruction encoder is trainable for all methods (optional second set).

Make the default run MODE-FROZEN-ENC so LoRA-Diffusion is compared apples-to-apples on PEFT parameters.

METHODS / BASELINES TO RUN (SINGLE-TASK AND MULTI-TASK)
Minimum set:
1) Full Fine-Tuning (all base model weights trainable; instruction encoder frozen in MODE-FROZEN-ENC)
2) Weight LoRA (standard LoRA on attention + MLP; specify rank r=64; same target modules across tasks)
3) Adapter layers (Houlsby-style bottleneck adapters; bottleneck dim=256)
4) BitFit (bias-only)
5) LoRA-Diffusion (trajectory adapters; step-adaptive ranks 64/32/8; k=2 modules)

Optional (if already easy in repo; do not spend days if not available):
6) A timestep-aware weight LoRA baseline (TC-LoRA-lite): add a simple gating of LoRA updates by timestep embedding, e.g. scale BA by s(t)=MLP(emb(t)) or a learned per-phase scalar. This addresses “timestep-aware baseline missing” without a full hypernetwork.

EXPERIMENT MATRIX
For each task in {SST2, QNLI, MRPC}:
- Single-task: run methods (1–5) with 5 seeds.
For Multi-task:
- Train a single joint model on SST2+QNLI+MRPC with:
  - LoRA-Diffusion multitask router (if implemented) OR a simple uniform mixture if router not stable.
  - For baselines, do multi-task training with the same parameterization (e.g., one shared adapter set) so comparisons are fair.
- Evaluate per task using that task’s instruction + metric rules.

DATA + PROMPTS (CONSISTENT FORMATTING)
- Use HuggingFace datasets for GLUE: glue/sst2, glue/qnli, glue/mrpc.
- Create a single consistent instruction template per task.
- Ensure max_seq_len is fixed (e.g., 128) across tasks unless MRPC needs 256; if different, document it and keep consistent across methods.

CODE CHANGES YOU MUST DO
1) Config system:
   - Add a unified YAML/JSON config for task, method, seed, mode (freeze/train encoder), steps, lr, T, batch sizes.
2) Training script:
   - Ensure every method uses the same data pipeline, same evaluation intervals, and same logging format.
   - Ensure seed sets torch/numpy/random and any CUDA determinism flags used in repo.
3) Evaluation:
   - Implement token-level denoising accuracy computation consistently for train/val/test.
   - Implement head-free GLUE metric evaluation using decoding + verbalizer.
   - Save per-run metrics to a structured file (metrics.json).
4) Parameter accounting:
   - Implement a single function that returns:
     - total params
     - trainable params
     - trainable params excluding instruction encoder
     - breakdown by component (base, encoder, lora/adapters/bitfit/trajectory)
   - Dump this into params.json per run.
5) Aggregation:
   - Create scripts/aggregate_results.py that reads run directories and outputs:
     - results_summary.csv
     - results_summary.json
     - LaTeX table snippets for the paper (one for token accuracy, one for standard GLUE metrics, one for params/efficiency).
6) Job launcher:
   - Provide a script that prints or writes all commands for:
     - single-task matrix
     - multi-task matrix
7) CONSISTENCY FIXES (REVIEWER CALLED OUT TABLE CONTRADICTIONS)
   - Remove/avoid mixing multiple metrics in the same “accuracy” column.
   - Every table row must explicitly state which metric it is.
   - Ensure token accuracy numbers cannot be confused with GLUE accuracy numbers.
   - Eliminate placeholders. If something is not measured, do not include it.

OSC / SLURM REQUIREMENTS (ADD THIS)
You must create TWO OSC SLURM templates and a launcher that generates job submissions:
A) slurm_single_task.slurm
   - Runs exactly one (task, method, seed, mode) experiment per job.
B) slurm_multitask.slurm
   - Runs exactly one (multitask setting, method, seed, mode) per job.

For speed, use as much GPU compute as OSC allows:
- Use 1 GPU per job, and set CPU cores to support data loading efficiently (e.g., 8–16 cores).
- Use multiple jobs in parallel across seeds/methods.
- Use GPU type A100 if available on OSC.
- Use `--cpus-per-task` and `--mem` appropriately so dataloaders do not bottleneck.
- Enable mixed precision (fp16/bf16) if supported and already stable in repo.
- Add environment setup section in the SLURM scripts (conda activate, module load, etc.).
- Each SLURM job must write logs to a unique file path that includes task/method/seed.

Deliver a submit script:
- scripts/submit_all.sh that submits all single-task jobs first, then multi-task jobs.
- Make it easy to throttle concurrency (e.g., user can comment out parts or set a max parallel cap).

DELIVERABLES (WHAT YOU OUTPUT TO ME)
A) A short list of files you modified/added with paths.
B) The exact commands to run locally (if possible) AND the exact `sbatch` commands for OSC:
   - all single-task jobs
   - all multi-task jobs
C) A quick “sanity checklist” showing:
   - seeds used
   - where metrics are saved
   - how to regenerate the final CSV/LaTeX tables

START BY INSPECTING THE REPO
- Identify existing training entrypoints and how LoRA-Diffusion is currently implemented.
- Identify how conditioning/instruction encoder is wired.
- Identify current metric computation and logging.
Then implement the changes above with minimal disruption.

Important: Keep compute reasonable:
- 5 seeds, fixed steps (e.g., 10k for SST2, 5k for QNLI/MRPC unless you justify otherwise).
- Use the same diffusion steps T as current best setting (likely T=100).
