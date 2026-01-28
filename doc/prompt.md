You are a senior ML engineer + research scientist. Build a COMPLETE, runnable codebase for the paper-style method “LoRA-Diffusion: Parameter-Efficient Fine-Tuning via Low-Rank Trajectory Decomposition” for discrete diffusion language models (masked diffusion / SEDD-style).

Goal
- Implement trajectory-level low-rank perturbation modules that add a learnable perturbation to the denoising update at each diffusion step.
- Support step-adaptive rank allocation (r=64/32/8 for early/mid/late steps) and step-adaptive scaling σ(t) (1.0/0.5/0.25).
- Train on multiple NLP tasks (classification/QA/reasoning/summarization/translation/generation) using Hugging Face datasets + evaluation.
- Run on OSC (Ohio Supercomputer Center) GPUs via SLURM scripts.

IMPORTANT OUTPUT REQUIREMENTS
1) Output a full repository structure with file tree and full contents for each file.
2) Code must run end-to-end with minimal changes (only user-specific paths).
3) Include a working Slurm submission script, environment setup instructions, and a README with exact commands.
4) Use Python 3.10+, PyTorch, and HuggingFace. If you use any extra libs (accelerate/deepspeed/wandb), include them in requirements and explain how to enable/disable.
5) Include unit tests or sanity checks for key components (adapter shapes, rank schedule, forward pass).
6) Provide clear comments and docstrings.

ASSUMPTIONS / MODEL CHOICE
- Use a discrete masked diffusion language model baseline (preferred: implement a simple masked diffusion transformer, or if easier, wrap an existing HF model as denoiser).
- Implement diffusion steps T=100.
- Forward process: masking corruption with schedule β_t (linear or cosine). x_t is tokens with [MASK] corruption.
- Reverse model predicts x0 distribution given x_t and t (and optional instruction conditioning c).
- Instruction conditioning: implement Enc(c) using a text encoder (e.g., small transformer or the same model encoder). Keep it simple but correct.

METHOD TO IMPLEMENT (LoRA-Diffusion)
At each denoising step:
x_{t-1} = f_base(x_t, t, c) + δ_t
where f_base is the frozen pretrained denoising network output (logits or predicted x0 distribution).
δ_t is the learned low-rank trajectory perturbation produced by k modules (k default=2):
δ_t = sum_{i=1..k} σ(t) * g_i(x_t, t, c)
g_i(x_t, t, c) = A_i(c) * ReLU(B_i([h_t; Emb(t)]))
- h_t is the hidden representation of x_t (token embeddings after transformer block stack, or choose a consistent representation).
- B_i: down-projection to rank r(t)
- A_i(c): up-projection back to hidden/logit space, instruction-conditioned:
A_i(c) = W_A^{(i)} + W_{A,cond}^{(i)} Enc(c)
Important: avoid illegal LaTeX-style notation in code; define A_i(c) as a function that returns a matrix or a linear operator that depends on instruction embedding.
Implementation detail: you can parameterize A_i(c) as a base up-projection plus a low-rank modulation from Enc(c), e.g.:
A_i(c) v = W_A v + (U_i Enc(c)) ⊙ v  (or a FiLM-style modulation) — but keep it faithful to “instruction-conditioned up-projection” and explain the design.
You must clearly document how A_i(c) is implemented and why it corresponds to the equation.

STEP-ADAPTIVE RANK SCHEDULE
- r(t)=64 for t>2T/3
- r(t)=32 for T/3 < t ≤ 2T/3
- r(t)=8 for t ≤ T/3
Implement this exactly.
Also implement σ(t)=1.0 / 0.5 / 0.25 for the same regions.

TRAINING OBJECTIVE
- Use denoising loss: negative log-likelihood / cross entropy of x0 tokens given x_t, t, c.
- Add optional rank regularization (nuclear norm surrogate): implement a practical approximation such as Frobenius penalty or spectral norm penalty (explain; true nuclear norm is expensive).
- Add optional orthogonality regularization between modules when k>1: ||W_A^T W_A||_F^2 across i≠j.
Make these toggles via config.

BASELINE METHODS (for comparison)
Provide code paths (switchable via config) to run:
1) Full fine-tuning (all parameters trainable)
2) Standard weight LoRA applied to attention/FFN projections (use peft library if needed)
3) Prefix tuning or adapter layers (minimal implementation is OK)
4) BitFit (bias-only)
5) Our LoRA-Diffusion
Make sure metrics are comparable.

DATASETS / TASKS
Implement at least 3 tasks end-to-end for demonstration, and design code to scale to 15:
- One classification task: SST-2 or AGNews
- One QA task: SQuAD (or a smaller QA set if needed)
- One summarization task: XSum or CNN/DM (can use a small subset)
Use HF datasets with caching.

EVALUATION
- For classification: accuracy
- For QA: F1/EM if possible, otherwise token-level F1
- For summarization: ROUGE-L (use evaluate library)
Also report:
- trainable parameter count and % of base
- peak GPU memory (use torch.cuda.max_memory_allocated)
- training time per step and total time
- adapter storage size
Save results to JSON/CSV.

EXPERIMENT MANAGEMENT
- Use a YAML config system (hydra or plain YAML) for all hyperparameters:
model size, T, schedule, ranks, σ(t), k, lr, batch size, grad accumulation, fp16/bf16, seed, dataset/task, max steps, eval frequency, etc.
- Add deterministic seeding.
- Add checkpointing: save base model pointer + adapter weights separately.
- Add ability to resume from checkpoint.

OSC / SLURM SUPPORT
Provide:
- slurm/train.slurm (single run)
- slurm/sweep_array.slurm (job array over tasks/seeds)
Include:
- module load commands (generic placeholders)
- conda env activation
- GPU request lines (e.g., --gres=gpu:1)
- logging to file
- how to set dataset cache and output dirs on OSC scratch

PERFORMANCE / MEMORY
- Must support fp16 or bf16 mixed precision.
- Must support gradient accumulation.
- Optionally support accelerate for multi-GPU (nice-to-have).
But ensure single-GPU works first.

DELIVERABLES
A) File tree
B) Full code for each file
C) README with:
   - install steps
   - download/prepare data
   - commands to run each baseline and LoRA-Diffusion
   - how to run on OSC with sbatch
D) A minimal example command that runs quickly on a small subset (for sanity check)
E) A “full experiment” command template

Style constraints
- Clean, production-quality code.
- No pseudo-code. Give real code.
- Use type hints.
- Use dataclasses for configs.
- Use logging (python logging).
- Provide clear error messages.

Start now. Output the repo content in the order: README, requirements, configs, core modules, training scripts, eval scripts, slurm scripts, tests.
