# GLUE Experiment Sanity Checklist (Instruction2.md)

## Seeds
- Exactly **5 seeds**: `[42, 43, 44, 45, 46]` (set in `configs/experiment_glue.yaml`, `scripts/run_experiments.py`, `scripts/run_multi_seed_experiments.sh`, `scripts/submit_all.sh`).

## Where metrics are saved
- **Per run**: `{output_dir}/` contains:
  - `run_info.json` – git commit, seed, task, method, command, config.
  - `params.json` – trainable_params_peft_only, trainable_params_total, breakdown, checkpoint_mb.
  - `training_summary.json` – final metrics, training_wall_clock_seconds, time_per_step_seconds.
  - `final_model/` – checkpoint (model.pt, lora_module.pt or instruction_encoder.pt).
- **After evaluation**: `eval_results.json`, `metrics.json` (token_accuracy_*, glue_accuracy_*, glue_f1_*).

## Regenerating CSV and LaTeX tables
1. Run evaluation on all run dirs (e.g. `scripts/run_multi_seed_experiments.sh` Step 2 with `--run_generation` for classification).
2. Run aggregation:
   ```bash
   python scripts/aggregate_results.py \
     --base_dir ./outputs/multi_seed_experiments \
     --tasks sst2 qnli mrpc \
     --methods full_ft lora_diffusion weight_lora adapters bitfit \
     --seeds 42-46 \
     --output_dir .
   ```
3. Outputs: `results_summary.csv`, `results_summary.json`, `table_token_accuracy.tex`, `table_glue_metrics.tex`, `table_params_efficiency.tex`.

## Local commands (no SLURM)
- Single run: `python scripts/train.py --config configs/base_config.yaml --task sst2 --method lora_diffusion --seed 42 --output_dir ./out/sst2_lora_seed42`
- Multi-seed: `python scripts/run_experiments.py --tasks sst2 qnli mrpc --methods full_ft lora_diffusion weight_lora adapters bitfit --seeds 42 43 44 45 46 --output_dir ./outputs/glue`
- Multitask joint: `python scripts/train_multitask_joint.py --method lora_diffusion --seed 42 --output_dir ./out/mt_lora_seed42`

## SLURM (OSC)
- Single-task: `TASK=sst2 METHOD=lora_diffusion SEED=42 ENCODER_MODE=frozen sbatch slurm/slurm_single_task.slurm`
- Multitask: `METHOD=lora_diffusion SEED=42 sbatch slurm/slurm_multitask.slurm`
- All: `bash scripts/submit_all.sh` (optionally `MAX_PARALLEL=20 bash scripts/submit_all.sh`).
