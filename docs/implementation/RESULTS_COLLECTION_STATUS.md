# Results Collection and Paper Table Update Status

## Summary

All paper tables have been updated with consistent parameter counts from `parameter_counts.json`. Validation accuracy values are pending completion of the OSC job (ID: 43996576) which is currently running.

## Completed Tasks ✓

1. **Parameter Counts Collected**: All methods have consistent parameter/storage numbers from `parameter_counts.json`
   - Full Fine-Tuning: 137.7M params (100.0%), 525.2 MB
   - LoRA-Diffusion: 39.6M params (28.7%), 150.9 MB
   - Weight LoRA: 9.7M params (6.6%), 36.9 MB
   - Adapters: 18.9M params (12.1%), 72.2 MB
   - BitFit: 156.5K params (0.11%), 0.6 MB
   - Prefix Tuning: 9.9M params (7.2%), 37.7 MB

2. **All Tables Updated**:
   - Table 3 (`tab:main_results`): Parameter percentages updated
   - Table 4 (`tab:per_task_results`): Parameter percentages and training steps updated
   - Table 5 (`tab:efficiency`): Parameter counts, percentages, and storage updated
   - Table 6 (`tab:forgetting`): Training loss data from full_ft available
   - Table 7 (`tab:composition`): Parameter percentages updated

3. **Parameter Discrepancy Handled**: Added note explaining that LoRA-Diffusion's 28.7% includes the instruction encoder (37.8M params, 27.5%), while trajectory adapters alone comprise 1.7M params (1.1%)

4. **Text References Updated**: Updated text to reflect new parameter counts

5. **Scripts Created**:
   - `scripts/collect_results.py`: Collects all experiment results
   - `scripts/update_paper_tables.py`: Updates paper tables with collected results
   - `scripts/monitor_and_update.py`: Monitors job and automatically collects/updates
   - `scripts/final_collect_and_update.sh`: Final script to run after job completion

## Pending Tasks ⏳

1. **Job Completion**: OSC job 43996576 is still running (currently training methods)
   - Step 1: Parameter counting ✓ (completed)
   - Step 2: SST-2 training (in progress)
   - Step 3: Evaluation (pending - will create `eval_results.json` files)
   - Step 4: Results summary (pending)

2. **Validation Accuracies**: Once evaluation completes, `eval_results.json` files will contain task-specific validation accuracies that need to be collected and added to tables

3. **Final Consistency Check**: Verify all numbers are consistent across tables once final results are available

## Next Steps

Once the job completes:

1. Run: `bash scripts/final_collect_and_update.sh 43996576 sst2`
   - This will collect all results including `eval_results.json` files
   - Update all tables with validation accuracies
   - Print summary

2. Or manually:
   ```bash
   python scripts/collect_results.py \
       --parameter-counts parameter_counts.json \
       --task sst2 \
       --job-id 43996576 \
       --output-dir lora-diffusion/outputs \
       --output collected_results_final.json
   
   python scripts/update_paper_tables.py \
       --collected-results collected_results_final.json \
       --paper-tex doc/Paper.tex \
       --handle-discrepancy
   ```

## Current Status

- **Job Status**: Running (check with `squeue -j 43996576`)
- **Tables Updated**: Parameter counts ✓, Validation accuracies pending
- **Paper File**: `doc/Paper.tex` (updated with parameter counts, ready for final accuracy values)

## Notes

- LoRA-Diffusion parameter count (28.7%) is higher than initially expected because it includes the instruction encoder. The note in the paper explains this breakdown.
- All parameter percentages are now consistent across all tables.
- Validation accuracy values currently show "---" and will be filled in once evaluation completes.
