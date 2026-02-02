# Experiment Job Submitted

## Job Information

- **Job ID**: 43996576
- **Script**: `slurm/run_experiments_osc.sh`
- **Submitted**: $(date)
- **Status**: Submitted to OSC queue

## What This Job Will Do

The job will run a complete set of experiments to address reviewer concerns:

### Step 1: Parameter Counting
- Runs `scripts/count_parameters.py` to generate consistent parameter/storage numbers
- Outputs to: `parameter_counts.json`
- Counts all methods: full_ft, lora_diffusion, weight_lora, adapters, bitfit, prefix_tuning

### Step 2: SST-2 Training Experiments
Trains all baseline methods on SST-2 with fixed evaluation:
- Full fine-tuning
- LoRA-Diffusion
- Weight LoRA (with MLP layers)
- Adapters
- BitFit

Each experiment:
- Uses seed 42
- Runs for MAX_STEPS (default: 5000 from config)
- Saves outputs to: `/fs/scratch/users/$USER/lora-diffusion/outputs/sst2_{method}_seed42_{job_id}/`
- Saves checkpoints to: `/fs/scratch/users/$USER/lora-diffusion/checkpoints/sst2_{method}_seed42_{job_id}/`

### Step 3: Evaluation
- Evaluates all trained models on validation set
- Uses fixed evaluation pipeline (text generation + label matching)
- Saves results to: `{output_dir}/eval_results.json`

### Step 4: Results Summary
- Collects all results into: `experiment_summary_{job_id}.json`
- Includes metrics for all methods

## Resource Allocation

- **Time**: 48 hours
- **GPUs**: 1 GPU
- **CPUs**: 8
- **Memory**: 128GB
- **Account**: pcs0229

## Monitoring the Job

### Check Status
```bash
squeue -j 43996576
```

### View Logs
```bash
# Output log
tail -f logs/slurm-43996576.out

# Error log  
tail -f logs/slurm-43996576.err
```

### Check Results (after completion)
```bash
# Parameter counts
cat parameter_counts.json

# Experiment summary
cat experiment_summary_43996576.json

# Individual method results
ls -lh /fs/scratch/users/$USER/lora-diffusion/outputs/sst2_*/
```

## Expected Outputs

After completion, you should have:

1. **parameter_counts.json** - Consistent parameter/storage numbers for all methods
2. **experiment_summary_43996576.json** - Summary of all experiment results
3. **Individual outputs** for each method:
   - Training logs
   - Evaluation results
   - Checkpoints
   - Metrics

## Next Steps After Job Completes

1. **Update paper tables** with consistent numbers from `parameter_counts.json`
2. **Review evaluation results** to verify baselines achieve credible performance
3. **Update paper** with new results if they differ significantly from previous numbers
4. **Run additional experiments** if needed (multi-task, ablations, etc.)

## Notes

- The job uses the fixed evaluation pipeline (generates text, computes task-specific metrics)
- All experiments use the same seed (42) for reproducibility
- Checkpoints are saved for later evaluation or analysis
- Results are saved to scratch space for efficiency
