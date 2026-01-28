"""Unit tests for data collators and evaluation metrics."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from modules directly to avoid loading task_loader (requires 'datasets')
import importlib.util
_collators_spec = importlib.util.spec_from_file_location(
    "collators",
    Path(__file__).parent.parent / "src" / "data" / "collators.py",
)
_collators = importlib.util.module_from_spec(_collators_spec)
_collators_spec.loader.exec_module(_collators)
DiffusionCollator = _collators.DiffusionCollator

from src.evaluation.metrics import (
    normalize_answer,
    compute_exact_match,
    compute_f1_score,
    compute_accuracy,
)


class MockTokenizer:
    """Mock tokenizer for collator tests."""

    pad_token_id = 0


class TestDiffusionCollator:
    """Test DiffusionCollator."""

    def test_collate_basic(self):
        """Test basic collation."""
        tokenizer = MockTokenizer()
        collator = DiffusionCollator(tokenizer=tokenizer, pad_to_multiple_of=None)

        features = [
            {
                "instruction_ids": [1, 2, 3],
                "instruction_mask": [1, 1, 1],
                "target_ids": [10, 20, 30, 40],
                "target_mask": [1, 1, 1, 1],
            },
            {
                "instruction_ids": [4, 5],
                "instruction_mask": [1, 1],
                "target_ids": [11, 21],
                "target_mask": [1, 1],
            },
        ]

        batch = collator(features)

        assert "instruction_ids" in batch
        assert "instruction_mask" in batch
        assert "target_ids" in batch
        assert "target_mask" in batch
        assert batch["instruction_ids"].shape == (2, 3)
        assert batch["target_ids"].shape == (2, 4)
        assert batch["instruction_ids"].dtype == torch.long
        assert batch["target_ids"].dtype == torch.long

        # First sample: no padding for instruction (max=3), second padded to 3
        assert batch["instruction_ids"][1][2] == tokenizer.pad_token_id
        assert batch["instruction_mask"][1][2] == 0
        # Targets: second padded to 4
        assert batch["target_ids"][1][2] == tokenizer.pad_token_id
        assert batch["target_mask"][1][2] == 0

    def test_collate_pad_to_multiple_of(self):
        """Test padding to multiple."""
        tokenizer = MockTokenizer()
        collator = DiffusionCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

        features = [
            {
                "instruction_ids": [1] * 5,
                "instruction_mask": [1] * 5,
                "target_ids": [1] * 10,
                "target_mask": [1] * 10,
            },
        ]

        batch = collator(features)
        assert batch["instruction_ids"].shape[1] == 8
        assert batch["target_ids"].shape[1] == 16

    def test_collate_with_task_labels(self):
        """Test collation with task labels."""
        tokenizer = MockTokenizer()
        collator = DiffusionCollator(tokenizer=tokenizer, pad_to_multiple_of=None)

        features = [
            {
                "instruction_ids": [1, 2],
                "instruction_mask": [1, 1],
                "target_ids": [10, 20],
                "target_mask": [1, 1],
                "task_label": 0,
            },
            {
                "instruction_ids": [3],
                "instruction_mask": [1],
                "target_ids": [11],
                "target_mask": [1],
                "task_label": 1,
            },
        ]

        batch = collator(features)
        assert "task_labels" in batch
        assert batch["task_labels"].shape == (2,)
        assert batch["task_labels"].tolist() == [0, 1]


class TestMetrics:
    """Test evaluation metrics."""

    def test_normalize_answer(self):
        """Test answer normalization."""
        assert normalize_answer("  Hello   World  ") == "hello world"
        assert normalize_answer("A Big Dog") == "big dog"
        assert "!" not in normalize_answer("hello!")

    def test_compute_exact_match(self):
        """Test exact match."""
        preds = ["hello world", "foo", "bar"]
        refs = ["hello world", "baz", "bar"]
        em = compute_exact_match(preds, refs)
        assert 0 <= em <= 1
        assert em == pytest.approx(2 / 3, rel=1e-5)

    def test_compute_f1_score(self):
        """Test token F1."""
        preds = ["hello world", "a b c"]
        refs = ["hello world", "a b d"]
        f1 = compute_f1_score(preds, refs)
        assert 0 <= f1 <= 1
        assert f1 > 0.5  # first exact, second partial overlap

    def test_compute_f1_empty(self):
        """Test F1 with empty tokens."""
        preds = ["", "x"]
        refs = ["", "x"]
        f1 = compute_f1_score(preds, refs)
        assert 0 <= f1 <= 1

    def test_compute_accuracy(self):
        """Test classification accuracy."""
        preds = ["positive", "negative", "positive"]
        refs = ["positive", "negative", "negative"]
        acc = compute_accuracy(preds, refs)
        assert acc == pytest.approx(2 / 3, rel=1e-5)

    def test_compute_accuracy_case_insensitive(self):
        """Test accuracy is case-insensitive."""
        preds = ["Positive"]
        refs = ["positive"]
        assert compute_accuracy(preds, refs) == pytest.approx(1.0, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
