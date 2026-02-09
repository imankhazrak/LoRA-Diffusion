# GLUE Jobs Status and Fix (Feb 2026)

## What we ran (yesterday / recent)

- **Single-task:** 75 jobs = 3 tasks (SST2, QNLI, MRPC) × 5 methods (full_ft, lora_diffusion, weight_lora, adapters, bitfit) × 5 seeds (42–46).
- **Multi-task:** 25 jobs = 5 methods × 5 seeds.

Submit commands used:
- Single-task only: `SINGLE_TASK_ONLY=1 bash scripts/submit_all.sh`
- Multi-task only: `MULTITASK_ONLY=1 bash scripts/submit_all.sh`

---

## Root cause of failures: disk quota exceeded

Almost all failures were due to **OSError: [Errno 122] Disk quota exceeded** (and related `basic_ios::clear: iostream error` when writing logs/checkpoints).

- Checkpoints and run outputs were written under the **project (home) directory**: `./outputs/glue_single/` and `./outputs/glue_multitask/`.
- Many jobs ran in parallel and filled the quota; once the disk quota was hit, any further write (checkpoint, log, tqdm refresh) caused the job to exit with code 1.

---

## Fix applied in SLURM scripts

1. **Outputs on scratch when available**  
   In both `slurm/slurm_single_task.slurm` and `slurm/slurm_multitask.slurm`:
   - If `/fs/scratch/users/$USER` exists and is writable, we set `SCRATCH` there.
   - When `SCRATCH` is not the project dir, **output directories are switched to scratch** so checkpoints and run dirs do not use home:
     - Single-task: `OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/glue_single`
     - Multi-task: `OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/glue_multitask`
   - Cache dirs (HF, transformers, torch) were already under `$SCRATCH` when scratch is used.

2. **Multitask script**  
   `slurm/slurm_multitask.slurm` already had the same SCRATCH/PROJECT_DIR fallback and cache setup as single-task; it now also redirects outputs to scratch when available.

After this change, re-submitted jobs will write **run dirs and checkpoints under**:
- `$SCRATCH/lora-diffusion/outputs/glue_single/`
- `$SCRATCH/lora-diffusion/outputs/glue_multitask/`

You can copy results back to the repo later, e.g.:
```bash
cp -r $SCRATCH/lora-diffusion/outputs/glue_single/* ./outputs/glue_single/
cp -r $SCRATCH/lora-diffusion/outputs/glue_multitask/* ./outputs/glue_multitask/
```

---

## Job outcome summary (from sacct)

- **Single-task (glue_st):**
  - **COMPLETED:** 30 jobs (44167435, 44167486–44167510, 44167519–44167520 show RUNNING in one run but many in that range completed before quota was hit).
  - **FAILED:** 46+ jobs (exit 1:0), including 44167386, 44167511–44167560, 44167619. Failures are due to disk quota (and downstream iostream errors).
- **Multi-task (glue_mt):**
  - **FAILED:** All 25 jobs (44167594–44167618), almost all after a few seconds, consistent with quota or inability to create run dirs under project.

Exact counts: run:
```bash
sacct -u $USER --format=JobID,JobName%14,State,ExitCode -n | grep -E "glue_st|glue_mt" | grep -v "\.ba+\|\.ex+"
```

---

## What to do next

1. **Re-submit after the fix**  
   With the updated SLURM scripts, new jobs will write to scratch and should avoid quota:
   - Single-task only: `SINGLE_TASK_ONLY=1 bash scripts/submit_all.sh`
   - Multi-task only: `MULTITASK_ONLY=1 bash scripts/submit_all.sh`
   - Or both: `bash scripts/submit_all.sh`

2. **Free space (optional)**  
   If you want to keep some runs under the project dir, reduce `outputs/` size (e.g. remove or archive old checkpoints), then you could run with scratch disabled for a test (not recommended for full 75+25 runs).

3. **Collect results after re-runs**  
   When jobs finish, aggregate token-level accuracy from:
   - `$SCRATCH/lora-diffusion/outputs/glue_single/<task>_<method>_seed<seed>_frozen/evaluation_history.json` (or `training_summary.json`)
   - Same for `glue_multitask/` if you re-run multi-task.

---

## Files changed

- `slurm/slurm_single_task.slurm`: when scratch is used, set `OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/glue_single`.
- `slurm/slurm_multitask.slurm`: when scratch is used, set `OUTPUT_DIR=$SCRATCH/lora-diffusion/outputs/glue_multitask`.
