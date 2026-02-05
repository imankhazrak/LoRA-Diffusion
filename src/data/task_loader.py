"""Task-specific data loaders for various NLP tasks."""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TaskExample:
    """Single task example."""
    input_text: str
    target_text: str
    instruction: str
    metadata: Dict[str, Any]


class TaskDataset(Dataset):
    """Generic task dataset."""
    
    def __init__(
        self,
        examples: List[TaskExample],
        tokenizer: AutoTokenizer,
        max_seq_length: int = 512,
        max_target_length: Optional[int] = None,
        task_name: Optional[str] = None,
        task_id: Optional[int] = None,
    ):
        """
        Args:
            examples: List of task examples
            tokenizer: Tokenizer
            max_seq_length: Maximum sequence length
            max_target_length: Maximum target length (for generation tasks)
            task_name: Name of the task (for multi-task composition)
            task_id: Integer ID of the task (for router training)
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_target_length = max_target_length or max_seq_length
        self.task_name = task_name
        self.task_id = task_id
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        # Tokenize instruction
        instruction_encoding = self.tokenizer(
            example.instruction,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # Tokenize target (the text to generate/predict)
        target_encoding = self.tokenizer(
            example.target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        result = {
            "instruction_ids": instruction_encoding["input_ids"],
            "instruction_mask": instruction_encoding["attention_mask"],
            "target_ids": target_encoding["input_ids"],
            "target_mask": target_encoding["attention_mask"],
            "input_text": example.input_text,
            "target_text": example.target_text,
            "metadata": example.metadata,
        }
        
        # Add task information for multi-task composition
        if self.task_name is not None:
            result["task_name"] = self.task_name
        if self.task_id is not None:
            result["task_label"] = self.task_id
        
        return result


class SST2Loader:
    """Loader for SST-2 sentiment classification."""
    
    @staticmethod
    def load(
        split: str,
        cache_dir: str,
        task_config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> List[TaskExample]:
        """Load SST-2 dataset."""
        import time
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # Try loading dataset
                dataset = load_dataset(
                    task_config["dataset"],
                    task_config.get("dataset_config"),
                    split=task_config["split_names"][split],
                    cache_dir=cache_dir,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to load dataset (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Last attempt: try without dataset_config (use default)
                    try:
                        logger.warning("Trying alternative dataset loading method...")
                        if task_config.get("dataset_config"):
                            dataset = load_dataset(
                                task_config["dataset"],
                                split=task_config["split_names"][split],
                                cache_dir=cache_dir,
                            )
                            # Filter to the specific config if needed
                            if hasattr(dataset, task_config["dataset_config"]):
                                dataset = dataset[task_config["dataset_config"]]
                        else:
                            raise
                    except Exception as e2:
                        logger.error(f"Failed to load dataset after {max_retries} attempts and fallback: {e2}")
                        raise e2
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        examples = []
        label_names = task_config.get("label_names", ["negative", "positive"])
        instruction_template = task_config.get(
            "instruction_template",
            "Classify the sentiment of the following sentence as positive or negative: {input}"
        )
        
        for item in dataset:
            # Handle different dataset formats
            input_text = item.get(task_config["input_column"], item.get("text", item.get("sentence", "")))
            label_id = item.get(task_config["label_column"], item.get("label", 0))
            # Handle both integer and string labels
            if isinstance(label_id, str):
                label_id = label_names.index(label_id) if label_id in label_names else int(label_id)
            label_text = label_names[label_id] if label_id < len(label_names) else str(label_id)
            
            instruction = instruction_template.format(input=input_text)
            
            examples.append(TaskExample(
                input_text=input_text,
                target_text=label_text,
                instruction=instruction,
                metadata={"label": label_id},
            ))
        
        logger.info(f"Loaded {len(examples)} SST-2 {split} examples")
        return examples


class SQuADLoader:
    """Loader for SQuAD question answering."""
    
    @staticmethod
    def load(
        split: str,
        cache_dir: str,
        task_config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> List[TaskExample]:
        """Load SQuAD dataset."""
        import time
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                dataset = load_dataset(
                    task_config["dataset"],
                    task_config.get("dataset_config"),
                    split=task_config["split_names"][split],
                    cache_dir=cache_dir,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to load dataset (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to load dataset after {max_retries} attempts: {e}")
                    raise
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        examples = []
        instruction_template = task_config.get(
            "instruction_template",
            "Answer the following question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"
        )
        
        for item in dataset:
            context = item[task_config["context_column"]]
            question = item[task_config["question_column"]]
            answers = item[task_config["answer_column"]]
            
            # Use first answer as target
            if isinstance(answers, dict) and "text" in answers:
                target_text = answers["text"][0] if answers["text"] else ""
            else:
                target_text = answers[0] if answers else ""
            
            instruction = instruction_template.format(
                context=context,
                question=question
            )
            
            examples.append(TaskExample(
                input_text=f"{context} {question}",
                target_text=target_text,
                instruction=instruction,
                metadata={
                    "context": context,
                    "question": question,
                    "answers": answers,
                },
            ))
        
        logger.info(f"Loaded {len(examples)} SQuAD {split} examples")
        return examples


class XSumLoader:
    """Loader for XSum summarization."""
    
    @staticmethod
    def load(
        split: str,
        cache_dir: str,
        task_config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> List[TaskExample]:
        """Load XSum dataset."""
        import time
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                dataset = load_dataset(
                    task_config["dataset"],
                    task_config.get("dataset_config"),
                    split=task_config["split_names"][split],
                    cache_dir=cache_dir,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to load dataset (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to load dataset after {max_retries} attempts: {e}")
                    raise
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        examples = []
        instruction_template = task_config.get(
            "instruction_template",
            "Summarize the following article in one sentence:\n{input}\nSummary:"
        )
        
        for item in dataset:
            input_text = item[task_config["input_column"]]
            target_text = item[task_config["target_column"]]
            
            instruction = instruction_template.format(input=input_text)
            
            examples.append(TaskExample(
                input_text=input_text,
                target_text=target_text,
                instruction=instruction,
                metadata={},
            ))
        
        logger.info(f"Loaded {len(examples)} XSum {split} examples")
        return examples


class MRPCLoader:
    """Loader for MRPC paraphrase classification (sentence1, sentence2 -> label)."""

    @staticmethod
    def load(
        split: str,
        cache_dir: str,
        task_config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> List[TaskExample]:
        """Load MRPC dataset."""
        import time
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                dataset = load_dataset(
                    task_config["dataset"],
                    task_config.get("dataset_config"),
                    split=task_config["split_names"][split],
                    cache_dir=cache_dir,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to load MRPC (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        examples = []
        label_names = task_config.get("label_names", ["not_equivalent", "equivalent"])
        instruction_template = task_config.get(
            "instruction_template",
            "Determine if the following two sentences are paraphrases. Sentence 1: {sentence1} Sentence 2: {sentence2} Answer (equivalent or not_equivalent):"
        )

        for item in dataset:
            sentence1 = item.get("sentence1", "")
            sentence2 = item.get("sentence2", "")
            label_id = item.get(task_config["label_column"], item.get("label", 0))
            if isinstance(label_id, str):
                label_id = label_names.index(label_id) if label_id in label_names else int(label_id)
            label_text = label_names[label_id] if label_id < len(label_names) else str(label_id)
            instruction = instruction_template.format(sentence1=sentence1, sentence2=sentence2)
            input_text = f"{sentence1} [SEP] {sentence2}"
            examples.append(TaskExample(
                input_text=input_text,
                target_text=label_text,
                instruction=instruction,
                metadata={"label": label_id},
            ))

        logger.info(f"Loaded {len(examples)} MRPC {split} examples")
        return examples


class QNLILoader:
    """Loader for QNLI (question-sentence NLI): question, sentence -> label."""

    @staticmethod
    def load(
        split: str,
        cache_dir: str,
        task_config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> List[TaskExample]:
        """Load QNLI dataset."""
        import time
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                dataset = load_dataset(
                    task_config["dataset"],
                    task_config.get("dataset_config"),
                    split=task_config["split_names"][split],
                    cache_dir=cache_dir,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to load QNLI (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        examples = []
        label_names = task_config.get("label_names", ["entailment", "not_entailment"])
        instruction_template = task_config.get(
            "instruction_template",
            "Does the sentence answer the question? Question: {question} Sentence: {sentence} Answer (entailment or not_entailment):"
        )

        for item in dataset:
            question = item.get("question", item.get("question", ""))
            sentence = item.get("sentence", item.get("sentence", ""))
            label_id = item.get(task_config["label_column"], item.get("label", 0))
            if isinstance(label_id, str):
                label_id = label_names.index(label_id) if label_id in label_names else int(label_id)
            label_text = label_names[label_id] if label_id < len(label_names) else str(label_id)
            instruction = instruction_template.format(question=question, sentence=sentence)
            input_text = f"{question} [SEP] {sentence}"
            examples.append(TaskExample(
                input_text=input_text,
                target_text=label_text,
                instruction=instruction,
                metadata={"label": label_id},
            ))

        logger.info(f"Loaded {len(examples)} QNLI {split} examples")
        return examples


# Registry of task loaders
TASK_LOADERS = {
    "sst2": SST2Loader,
    "squad": SQuADLoader,
    "xsum": XSumLoader,
    "mrpc": MRPCLoader,
    "qnli": QNLILoader,
}


def get_task_loader(
    task_name: str,
    split: str,
    cache_dir: str,
    task_config: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_samples: Optional[int] = None,
    max_seq_length: int = 512,
) -> TaskDataset:
    """
    Get task dataset.
    
    Args:
        task_name: Name of the task
        split: Dataset split ("train", "validation", "test")
        cache_dir: Cache directory for datasets
        task_config: Task configuration
        tokenizer: Tokenizer
        max_samples: Maximum number of samples to load
        max_seq_length: Maximum sequence length
        
    Returns:
        TaskDataset instance
    """
    task_name_lower = task_name.lower()
    
    if task_name_lower not in TASK_LOADERS:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASK_LOADERS.keys())}"
        )
    
    loader_class = TASK_LOADERS[task_name_lower]
    examples = loader_class.load(
        split=split,
        cache_dir=cache_dir,
        task_config=task_config,
        max_samples=max_samples,
    )
    
    max_target_length = task_config.get("max_target_length", max_seq_length)
    
    # Get task ID from config if available (for multi-task training)
    task_id = task_config.get("task_id", None)
    
    return TaskDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_target_length=max_target_length,
        task_name=task_name,
        task_id=task_id,
    )
