"""Data collators for batching diffusion model inputs."""

from typing import List, Dict, Any
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DiffusionCollator:
    """
    Collator for diffusion model training.
    
    Handles padding and batching of instruction and target sequences.
    """
    
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: int = 8
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            features: List of examples from TaskDataset
            
        Returns:
            Batched tensors
        """
        # Extract and pad instruction sequences
        instruction_ids = [f["instruction_ids"] for f in features]
        instruction_mask = [f["instruction_mask"] for f in features]
        
        max_inst_len = max(len(ids) for ids in instruction_ids)
        if self.pad_to_multiple_of:
            max_inst_len = (
                (max_inst_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        
        padded_instruction_ids = []
        padded_instruction_mask = []
        
        for ids, mask in zip(instruction_ids, instruction_mask):
            pad_len = max_inst_len - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * pad_len
            padded_mask = mask + [0] * pad_len
            padded_instruction_ids.append(padded_ids)
            padded_instruction_mask.append(padded_mask)
        
        # Extract and pad target sequences
        target_ids = [f["target_ids"] for f in features]
        target_mask = [f["target_mask"] for f in features]
        
        max_target_len = max(len(ids) for ids in target_ids)
        if self.pad_to_multiple_of:
            max_target_len = (
                (max_target_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        
        padded_target_ids = []
        padded_target_mask = []
        
        for ids, mask in zip(target_ids, target_mask):
            pad_len = max_target_len - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * pad_len
            padded_mask = mask + [0] * pad_len
            padded_target_ids.append(padded_ids)
            padded_target_mask.append(padded_mask)
        
        # Convert to tensors
        batch = {
            "instruction_ids": torch.tensor(padded_instruction_ids, dtype=torch.long),
            "instruction_mask": torch.tensor(padded_instruction_mask, dtype=torch.long),
            "target_ids": torch.tensor(padded_target_ids, dtype=torch.long),
            "target_mask": torch.tensor(padded_target_mask, dtype=torch.long),
        }
        
        # Optionally include text for evaluation
        if "input_text" in features[0]:
            batch["input_texts"] = [f["input_text"] for f in features]
            batch["target_texts"] = [f["target_text"] for f in features]
        
        if "metadata" in features[0]:
            batch["metadata"] = [f["metadata"] for f in features]
        
        # Include task labels for multi-task composition
        if "task_label" in features[0]:
            task_labels = [f["task_label"] for f in features]
            batch["task_labels"] = torch.tensor(task_labels, dtype=torch.long)
        
        if "task_name" in features[0]:
            batch["task_names"] = [f["task_name"] for f in features]
        
        return batch
