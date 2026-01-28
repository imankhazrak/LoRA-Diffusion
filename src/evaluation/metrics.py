"""Metrics computation for various NLP tasks."""

import logging
from typing import List, Dict, Any, Callable
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


def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute classification accuracy."""
    correct = sum(
        pred.strip().lower() == ref.strip().lower()
        for pred, ref in zip(predictions, references)
    )
    return correct / len(predictions)


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
        def classification_metrics(predictions, references):
            return {
                "accuracy": compute_accuracy(predictions, references),
            }
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
) -> Dict[str, float]:
    """
    Compute task-specific metrics.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        task_config: Task configuration dictionary
        
    Returns:
        Dictionary of computed metrics
    """
    task_type = task_config["type"]
    metric_fn = get_metric_fn(task_type)
    
    return metric_fn(predictions, references)
