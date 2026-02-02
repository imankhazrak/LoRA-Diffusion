# Why 80% in the First Experiment vs ~50% Now?

## Short Answer

**The ~80% was training (token-level) accuracy. The ~50% is validation (task-level) accuracy from text generation.** They measure different things.

## Two Different Metrics

### 1. Training accuracy (~80%) — what you saw first

- **Where it comes from**: During training, the trainer computes accuracy **inside the training step**.
- **How it's computed**: For each **masked token position**, the model predicts the correct token (e.g. "positive" or "negative"). Accuracy = (correct token predictions) / (number of masked positions).
- **Code**: In `src/models/base_diffusion.py`, the `forward_diffusion_loss()` method does:
  ```python
  pred_tokens = logits.argmax(dim=-1)
  correct = (pred_tokens == x0) & mask
  accuracy = correct.float().sum() / mask.float().sum()
  ```
- **What it means**: "On the **training set**, when we mask some tokens and ask the model to predict them, it gets ~83% of those tokens right." This is **token-level accuracy on the training data**.
- **Why it's high**: The model is trained on the same examples repeatedly and only has to predict a few tokens (often the label token). So it can reach ~80–85% on this metric.

### 2. Validation accuracy (~50%) — what we report now

- **Where it comes from**: The **evaluation script** (`scripts/evaluate.py`) runs **full reverse diffusion** to generate text from scratch, then decodes the text and matches it to labels ("positive" / "negative").
- **How it's computed**: For each validation example, we (1) start from noise/mask, (2) run 100 denoising steps to generate a sequence, (3) decode to text (e.g. "positive positive"), (4) map that text to a label (e.g. "positive"), (5) compare to the true label.
- **What it means**: "On the **validation set**, when we generate text from scratch and then interpret it as a label, we get the right label ~50% of the time." This is **task-level (classification) accuracy**.
- **Why it's ~50%**: The model often generates repetitive or noisy text (e.g. "positive positive"). Our decoder still maps that to "positive", but the model is not reliably generating the *correct* label for each input, so we end up near chance (~50% for balanced SST-2).

## Evidence from Your Runs

From `collected_results_improved.json` (and similar files):

| Method          | Final **train** acc (token-level) | **Val** acc (text gen → label) |
|-----------------|------------------------------------|---------------------------------|
| Full FT         | **83.5%**                          | 51.1%                           |
| LoRA-Diffusion  | **84.1%**                          | 49.0%                           |
| Weight LoRA     | **84.8%**                          | 48.6%                           |
| BitFit          | **83.2%**                          | 48.3%                           |

So:

- The **~80%** number is the **training (token-level) accuracy** (e.g. in `paper_results.json`: "Final Train Acc (%)": 80.96).
- The **~50%** number is the **validation (task-level) accuracy** from the evaluation script.

## Why the first experiment showed 80%

In the first experiment, the numbers that were shown (e.g. in logs, tables, or `paper_results.json`) were very likely:

- **Final Train Acc (%)** (token-level on training data), and/or  
- Metrics from the **training loop** (same token-level accuracy).

So the "80%" was never the same as "validation accuracy from full generation." It was training accuracy. Once we started reporting the **validation accuracy from the evaluation script** (generate → decode → match to label), we correctly got ~50%.

## Summary

| Metric              | ~80% (first experiment)     | ~50% (current)                    |
|---------------------|-----------------------------|-----------------------------------|
| What                | Training token-level acc    | Validation task-level acc         |
| When                | During training             | After training, in evaluate.py    |
| How                 | Predict masked tokens       | Full generation → decode → label   |
| Same thing?         | No                          | No                                |

So we did **not** "lose" 30% accuracy. We are now reporting a **different**, **correct** metric for the paper: **validation accuracy from text generation**, which is ~50%. The 80% was training (token-level) accuracy and should not be reported as the main classification result. For the paper, the right number to report is the ~50% validation accuracy, with a clear explanation that it comes from the generative evaluation protocol (and its limitations).
