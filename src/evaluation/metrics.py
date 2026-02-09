"""Metrics computation for various NLP tasks."""

import logging
from typing import List, Dict, Any, Callable, Optional
from collections import Counter
import string

import numpy as np

logger = logging.getLogger(__name__)

# Try to import evaluation libraries
try:
    from evaluate import load as load_metric
    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False
    logger.warning("evaluate library not found. Some metrics may not be available.")


def normalize_answer(s: str) -> str:
    """Normalize answer string for QA evaluation."""
    def remove_articles(text):
        return " ".join([w for w in text.split() if w.lower() not in ["a", "an", "the"]])
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute exact match score."""
    em_scores = [
        float(normalize_answer(pred) == normalize_answer(ref))
        for pred, ref in zip(predictions, references)
    ]
    return np.mean(em_scores)


def compute_f1_score(predictions: List[str], references: List[str]) -> float:
    """Compute token-level F1 score for QA."""
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(float(pred_tokens == ref_tokens))
            continue
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1_scores.append(0.0)
            continue
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def decode_classification_label(
    text: str,
    label_names: List[str],
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Decode generated text to a classification label.
    
    For classification tasks, the model may generate text like "positive" or "negative".
    This function tries to match the generated text to one of the valid labels,
    handling cases where the model generates extra words or variations.
    
    For diffusion models that generate sequences, we prioritize the first word/token
    since that's most likely to be the intended label.
    
    Args:
        text: Generated text
        label_names: List of valid label names (e.g., ["negative", "positive"])
        tokenizer: Optional tokenizer for token-level matching
        
    Returns:
        Matched label name, or first label if no match found
    """
    if not text or not label_names:
        return label_names[0] if label_names else ""
    
    text_lower = text.strip().lower()
    
    # Token-level matching if tokenizer is provided
    if tokenizer is not None:
        try:
            # Get tokens from generated text
            text_tokens = tokenizer.encode(text, add_special_tokens=False)
            text_token_ids = set(text_tokens)
            
            # Check each label
            for label in label_names:
                label_tokens = tokenizer.encode(label, add_special_tokens=False)
                # Check if all label tokens appear in generated text (in order)
                if len(label_tokens) > 0:
                    # Check if label tokens appear as a contiguous sequence
                    for i in range(len(text_tokens) - len(label_tokens) + 1):
                        if text_tokens[i:i+len(label_tokens)] == label_tokens:
                            return label
                    # Also check if all label token IDs are present
                    if set(label_tokens).issubset(text_token_ids):
                        return label
        except Exception:
            # Fall back to word-level matching if tokenization fails
            pass
    
    # For classification, try to extract just the first word (most reliable)
    first_word = text_lower.split()[0] if text_lower.split() else ""
    
    # Try exact match with first word (highest priority)
    for label in label_names:
        if first_word == label.lower():
            return label
    
    # Try exact match with full text
    for label in label_names:
        if text_lower == label.lower():
            return label
    
    # Try prefix match (e.g., "pos" matches "positive")
    for label in label_names:
        label_lower = label.lower()
        if first_word.startswith(label_lower) or label_lower.startswith(first_word):
            return label
    
    # Try substring match with first word (e.g., "positive" in "positive negative...")
    for label in label_names:
        if label.lower() in first_word or first_word in label.lower():
            return label
    
    # Try substring match with full text (e.g., "the sentiment is positive" -> "positive")
    for label in label_names:
        if label.lower() in text_lower or text_lower in label.lower():
            return label
    
    # Try word-level match (check if any label word appears in the text)
    text_words = set(text_lower.split())
    for label in label_names:
        if label.lower() in text_words:
            return label
    
    # Try character-level similarity (for typos or variations)
    # Simple Levenshtein-like check: if first word is very similar to a label
    for label in label_names:
        label_lower = label.lower()
        # Check if first word and label share significant overlap
        if len(first_word) >= 3 and len(label_lower) >= 3:
            # Check if they share at least 70% of characters
            common_chars = sum(1 for c in first_word if c in label_lower)
            if common_chars >= min(len(first_word), len(label_lower)) * 0.7:
                return label
    
    # Default: return first label (or could return empty string)
    return label_names[0] if label_names else ""


def classification_head_accuracy(
    logits: np.ndarray,
    label_indices: np.ndarray,
) -> float:
    """
    Compute accuracy from classification-head logits and integer label indices.
    Used for standard classification-head evaluation (primary metric).
    
    Args:
        logits: (N, num_classes) logits from linear head
        label_indices: (N,) integer class indices (0, 1, ...)
        
    Returns:
        Accuracy in [0, 1].
    """
    pred = np.argmax(logits, axis=-1)
    return float(np.mean(pred == label_indices))


def compute_classification_f1(
    predictions: List[str],
    references: List[str],
    task_config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
) -> float:
    """Compute binary/macro F1 for classification (e.g. MRPC). Uses same decoding as accuracy."""
    if not predictions or not references:
        return 0.0
    label_names = None
    if task_config:
        if "task" in task_config and "label_names" in task_config["task"]:
            label_names = task_config["task"]["label_names"]
        elif "label_names" in task_config:
            label_names = task_config["label_names"]
    if not label_names:
        label_names = list(set(references))
    decoded = [
        decode_classification_label(p, label_names, tokenizer=tokenizer)
        for p in predictions
    ]
    refs_normalized = [r.strip().lower() for r in references]
    preds_normalized = [d.strip().lower() for d in decoded]
    # Binary F1: treat index 1 as positive class (e.g. "equivalent", "positive")
    pos_label = label_names[1].lower() if len(label_names) > 1 else refs_normalized[0]
    tp = sum(1 for p, r in zip(preds_normalized, refs_normalized) if p == pos_label and r == pos_label)
    fp = sum(1 for p, r in zip(preds_normalized, refs_normalized) if p == pos_label and r != pos_label)
    fn = sum(1 for p, r in zip(preds_normalized, refs_normalized) if p != pos_label and r == pos_label)
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_accuracy(
    predictions: List[str],
    references: List[str],
    task_config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        task_config: Optional task config with label_names for robust decoding
        tokenizer: Optional tokenizer for token-level label decoding
    """
    # If task_config provides label_names, use robust decoding
    if task_config:
        # Handle both formats: task_config["task"]["label_names"] or task_config["label_names"]
        label_names = None
        if "task" in task_config and "label_names" in task_config["task"]:
            label_names = task_config["task"]["label_names"]
        elif "label_names" in task_config:
            label_names = task_config["label_names"]
        
        # Try to get tokenizer from task_config if not provided
        if tokenizer is None and "tokenizer" in task_config:
            tokenizer = task_config["tokenizer"]
        
        if label_names:
            decoded_predictions = [
                decode_classification_label(pred, label_names, tokenizer=tokenizer)
                for pred in predictions
            ]
            correct = sum(
                pred.strip().lower() == ref.strip().lower()
                for pred, ref in zip(decoded_predictions, references)
            )
        else:
            # Simple string matching
            correct = sum(
                pred.strip().lower() == ref.strip().lower()
                for pred, ref in zip(predictions, references)
            )
    else:
        # Simple string matching
        correct = sum(
            pred.strip().lower() == ref.strip().lower()
            for pred, ref in zip(predictions, references)
        )
    return correct / len(predictions) if predictions else 0.0


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    if not HAS_EVALUATE:
        logger.warning("evaluate library not available. Skipping ROUGE computation.")
        return {"rouge_l": 0.0}
    
    try:
        rouge = load_metric("rouge")
        results = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        return {
            "rouge_1": results["rouge1"],
            "rouge_2": results["rouge2"],
            "rouge_l": results["rougeL"],
        }
    except Exception as e:
        logger.error(f"Error computing ROUGE: {e}")
        return {"rouge_l": 0.0}


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score."""
    if not HAS_EVALUATE:
        logger.warning("evaluate library not available. Skipping BLEU computation.")
        return 0.0
    
    try:
        bleu = load_metric("bleu")
        # Convert references to list of lists for BLEU
        refs = [[ref] for ref in references]
        results = bleu.compute(predictions=predictions, references=refs)
        return results["bleu"]
    except Exception as e:
        logger.error(f"Error computing BLEU: {e}")
        return 0.0


def get_metric_fn(task_type: str) -> Callable:
    """
    Get metric computation function for task type.
    
    Args:
        task_type: Type of task (classification, qa, summarization, generation)
        
    Returns:
        Function that computes metrics
    """
    if task_type == "classification":
        def classification_metrics(predictions, references, task_config=None, tokenizer=None):
            out = {
                "accuracy": compute_accuracy(predictions, references, task_config=task_config, tokenizer=tokenizer),
            }
            # MRPC and tasks with glue_metrics include F1 (task_config may be full config with config["metrics"])
            metrics_cfg = task_config.get("metrics", {}) if task_config else {}
            if metrics_cfg.get("glue_metrics") and "f1" in metrics_cfg.get("glue_metrics", []):
                out["f1"] = compute_classification_f1(predictions, references, task_config=task_config, tokenizer=tokenizer)
            return out
        return classification_metrics
    
    elif task_type == "qa":
        def qa_metrics(predictions, references):
            return {
                "exact_match": compute_exact_match(predictions, references),
                "f1": compute_f1_score(predictions, references),
            }
        return qa_metrics
    
    elif task_type == "summarization":
        def summarization_metrics(predictions, references):
            return compute_rouge(predictions, references)
        return summarization_metrics
    
    elif task_type == "generation":
        def generation_metrics(predictions, references):
            metrics = compute_rouge(predictions, references)
            metrics["bleu"] = compute_bleu(predictions, references)
            return metrics
        return generation_metrics
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def compute_metrics(
    predictions: List[str],
    references: List[str],
    task_config: Dict[str, Any],
    tokenizer: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Compute task-specific metrics.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        task_config: Task configuration dictionary (should contain "task" key with task config)
        tokenizer: Optional tokenizer for token-level label decoding
        
    Returns:
        Dictionary of computed metrics
    """
    # Handle both old format (task_config["type"]) and new format (task_config["task"]["type"])
    if "type" in task_config:
        task_type = task_config["type"]
        task_subconfig = task_config
    elif "task" in task_config:
        task_type = task_config["task"]["type"]
        task_subconfig = task_config
    else:
        raise ValueError("task_config must contain 'type' or 'task.type'")
    
    metric_fn = get_metric_fn(task_type)
    
    # For classification, pass task_config and tokenizer for robust label decoding
    if task_type == "classification":
        return metric_fn(predictions, references, task_config=task_subconfig, tokenizer=tokenizer)
    else:
        return metric_fn(predictions, references)
