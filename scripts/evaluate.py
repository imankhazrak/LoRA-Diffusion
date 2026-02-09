#!/usr/bin/env python3
"""Evaluation script for trained models."""

import argparse
import logging
from pathlib import Path
import sys
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MaskedDiffusionTransformer, LoRADiffusionModule, MultiTaskLoRAComposer
from src.models.lora_modules import InstructionEncoder
from src.data import get_task_loader, DiffusionCollator
from src.evaluation import compute_metrics
from src.training.losses import compute_diffusion_loss
from src.utils import load_config, setup_logging, set_seed
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LoRA-Diffusion model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--use_composition",
        action="store_true",
        help="Use multi-task composition (requires --task_modules and --task_names)",
    )
    parser.add_argument(
        "--task_modules",
        type=str,
        nargs="+",
        default=None,
        help="Paths to checkpoint directories for task-specific LoRA modules (for composition)",
    )
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="+",
        default=None,
        help="Names of tasks corresponding to task_modules (for composition)",
    )
    parser.add_argument(
        "--composition_mode",
        type=str,
        default="router",
        choices=["router", "uniform", "task_arithmetic"],
        help="Composition mode: router (default), uniform (1/M), or task_arithmetic (sum of deltas)",
    )
    parser.add_argument(
        "--eval_classification_head",
        action="store_true",
        help="Evaluate using classification head on final denoised representation (primary metric).",
    )
    parser.add_argument(
        "--teacher_forced",
        action="store_true",
        default=True,
        help="Use teacher-forced evaluation (token-level accuracy, matches training). Default for classification.",
    )
    parser.add_argument(
        "--no_teacher_forced",
        action="store_false",
        dest="teacher_forced",
        help="Disable teacher-forced eval; use generation instead.",
    )
    parser.add_argument(
        "--run_generation",
        action="store_true",
        help="Also run generation and save generation_accuracy (default: only token-level accuracy, same as training).",
    )
    
    return parser.parse_args()


@torch.no_grad()
def compute_teacher_forced_metrics(
    model,
    lora_module,
    dataloader,
    device: str,
    config: Dict[str, Any],
    instruction_encoder=None,
    instruction_to_hidden=None,
) -> Dict[str, float]:
    """
    Compute teacher-forced metrics (token-level accuracy) - same metric and aggregation as training.
    Uses compute_diffusion_loss forward pass without backprop.
    Aggregation: per-batch average (mean over batches), same as training log.
    """
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    for batch in tqdm(dataloader, desc="Teacher-forced eval"):
        loss, metrics = compute_diffusion_loss(
            model=model,
            batch=batch,
            lora_module=lora_module,
            config=config,
            instruction_encoder=instruction_encoder,
            instruction_to_hidden=instruction_to_hidden,
        )
        total_loss += loss.item()
        total_acc += metrics["accuracy"]
        num_batches += 1
    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "accuracy": total_acc / n,
    }


def find_config(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Find and load config from various possible locations."""
    # Try direct config.json
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    
    # Try training_summary.json
    summary_path = checkpoint_path / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            if "config" in summary:
                logger.info(f"Found config in {summary_path}")
                return summary["config"]
    
    # Try checkpoint subdirectories
    for subdir_name in ["final_model", "best_model"]:
        subdir = checkpoint_path / subdir_name
        config_path = subdir / "config.json"
        if config_path.exists():
            logger.info(f"Found config in {config_path}")
            with open(config_path) as f:
                return json.load(f)
    
    # Try latest checkpoint-*
    checkpoint_dirs = sorted(
        checkpoint_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
    )
    for ckpt_dir in reversed(checkpoint_dirs):
        config_path = ckpt_dir / "config.json"
        if config_path.exists():
            logger.info(f"Found config in {config_path}")
            with open(config_path) as f:
                return json.load(f)
    
    # Try looking in checkpoints/ directory relative to project root
    # Extract experiment name from checkpoint_path (e.g., sst2_full_ft_seed42)
    project_root = Path(__file__).parent.parent
    checkpoints_base = project_root / "checkpoints"
    if checkpoints_base.exists():
        # Try to find matching checkpoint directory
        exp_name = checkpoint_path.name
        for ckpt_dir in checkpoints_base.glob(f"{exp_name}*"):
            for subdir_name in ["final_model", "best_model"]:
                subdir = ckpt_dir / subdir_name
                config_path = subdir / "config.json"
                if config_path.exists():
                    logger.info(f"Found config in {config_path}")
                    with open(config_path) as f:
                        return json.load(f)
            # Also try checkpoint-* subdirectories
            checkpoint_subdirs = sorted(
                ckpt_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
            )
            for ckpt_subdir in reversed(checkpoint_subdirs):
                config_path = ckpt_subdir / "config.json"
                if config_path.exists():
                    logger.info(f"Found config in {config_path}")
                    with open(config_path) as f:
                        return json.load(f)
    
    return None


def find_checkpoint_subdir(checkpoint_path: Path, config: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Find the actual checkpoint subdirectory containing model files."""
    # Try final_model
    final_model = checkpoint_path / "final_model"
    if (final_model / "model.pt").exists() or (final_model / "lora_module.pt").exists():
        logger.info(f"Found checkpoint in {final_model}")
        return final_model
    
    # Try best_model
    best_model = checkpoint_path / "best_model"
    if (best_model / "model.pt").exists() or (best_model / "lora_module.pt").exists():
        logger.info(f"Found checkpoint in {best_model}")
        return best_model
    
    # Try latest checkpoint-*
    checkpoint_dirs = sorted(
        checkpoint_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
    )
    for ckpt_dir in reversed(checkpoint_dirs):
        if (ckpt_dir / "model.pt").exists() or (ckpt_dir / "lora_module.pt").exists():
            logger.info(f"Found checkpoint in {ckpt_dir}")
            return ckpt_dir
    
    # Fallback: try checkpoint_path itself
    if (checkpoint_path / "model.pt").exists() or (checkpoint_path / "lora_module.pt").exists():
        logger.info(f"Found checkpoint in {checkpoint_path}")
        return checkpoint_path
    
    # Do NOT fall back to checkpoints/ for multi-seed experiment dirs: that would load
    # a single shared checkpoint for all seeds and produce identical (wrong) val accuracy.
    # BitFit works because its experiment dirs had final_model/ when eval ran (from rerun);
    # full_ft/lora_diffusion had no final_model in experiment dirs, so fallback returned the
    # same checkpoint for every seed. Fail loudly instead.
    try:
        path_str = checkpoint_path.resolve().as_posix()
        if "multi_seed_experiments" in path_str and "_seed" in checkpoint_path.name:
            logger.warning(
                "Checkpoint not found in %s (no final_model/ or model.pt). "
                "Skipping checkpoints/ fallback for multi-seed experiment dirs to avoid "
                "loading the same checkpoint for all seeds (wrong val accuracy).",
                checkpoint_path,
            )
            return None
    except (OSError, RuntimeError):
        pass
    
    # Try looking in checkpoints/ directory (from config or default location)
    project_root = Path(__file__).parent.parent
    if config and "output" in config and "checkpoint_dir" in config["output"]:
        checkpoints_base = Path(config["output"]["checkpoint_dir"])
        if not checkpoints_base.is_absolute():
            checkpoints_base = project_root / checkpoints_base
    else:
        checkpoints_base = project_root / "checkpoints"
    
    if not checkpoints_base.exists():
        return None
    
    # Extract task and method from checkpoint_path name (e.g., "sst2_adapters_seed42" -> "sst2_adapters")
    exp_name = checkpoint_path.name
    # Remove "_seedXX" suffix if present
    if "_seed" in exp_name:
        base_pattern = exp_name.rsplit("_seed", 1)[0]
    else:
        base_pattern = exp_name
    
    # Get seed from config if available
    target_seed = None
    if config and "training" in config and "seed" in config["training"]:
        target_seed = config["training"]["seed"]
    
    # Search for checkpoint directories matching task_method pattern
    # Checkpoint dirs are named like "sst2_adapters_<jobid>"
    for ckpt_dir in checkpoints_base.glob(f"{base_pattern}_*"):
        if not ckpt_dir.is_dir():
            continue
        
        # Try to match by seed if we have it
        if target_seed is not None:
            # Check config.json in final_model or best_model to see if seed matches
            for subdir_name in ["final_model", "best_model"]:
                subdir = ckpt_dir / subdir_name
                config_path = subdir / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            ckpt_config = json.load(f)
                        ckpt_seed = ckpt_config.get("training", {}).get("seed")
                        if ckpt_seed == target_seed:
                            # Found matching seed, check if model files exist
                            if (subdir / "model.pt").exists() or (subdir / "lora_module.pt").exists():
                                logger.info(f"Found checkpoint in {subdir} (matched seed {target_seed})")
                                return subdir
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        # If no seed match or no seed available, try all subdirectories
        for subdir_name in ["final_model", "best_model"]:
            subdir = ckpt_dir / subdir_name
            if (subdir / "model.pt").exists() or (subdir / "lora_module.pt").exists():
                logger.info(f"Found checkpoint in {subdir} (no seed matching)")
                return subdir
        
        # Try latest checkpoint-* in this directory
        checkpoint_subdirs = sorted(
            ckpt_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
        )
        for ckpt_subdir in reversed(checkpoint_subdirs):
            if (ckpt_subdir / "model.pt").exists() or (ckpt_subdir / "lora_module.pt").exists():
                logger.info(f"Found checkpoint in {ckpt_subdir}")
                return ckpt_subdir
    
    return None


@torch.no_grad()
def generate_predictions(
    model,
    lora_module,
    composer,
    dataloader,
    device,
    config,
    use_composition=False,
    instruction_encoder=None,
    instruction_to_hidden=None,
):
    """Generate predictions for all samples."""
    model.eval()
    if lora_module:
        lora_module.eval()
    if composer:
        composer.eval()
    
    all_predictions = []
    all_references = []
    
    # Get tokenizer and task config
    tokenizer = dataloader.dataset.tokenizer
    task_type = config.get("task", {}).get("type", None)
    task_config = config.get("task", {})
    label_names = task_config.get("label_names", None)
    is_classification = task_type == "classification"
    
    # For classification, use fixed short sequence length (3-5 tokens)
    # This is enough for single-word labels like "positive" or "negative"
    classification_seq_len = 5
    
    for batch in tqdm(dataloader, desc="Generating"):
        target_texts = batch.get("target_texts", [])
        all_references.extend(target_texts)
        
        batch_size = batch["target_ids"].size(0)
        
        # Determine sequence length
        if is_classification:
            seq_len = classification_seq_len
        else:
            seq_len = batch["target_ids"].size(1)
        
        instruction_ids = batch["instruction_ids"].to(device)
        instruction_mask = batch["instruction_mask"].to(device)
        
        if use_composition and composer:
            # Use composition sampling
            # For classification, use greedy sampling
            if is_classification:
                # Get instruction embedding for composition
                if lora_module:
                    instruction_emb = lora_module.instruction_encoder(
                        instruction_ids,
                        attention_mask=instruction_mask,
                    )
                    instruction_emb = lora_module.instruction_to_hidden(instruction_emb)
                else:
                    instruction_emb = None
                
                samples = model.sample(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    instruction_embedding=instruction_emb,
                    device=device,
                    greedy=True,  # Use greedy for classification
                )
            else:
                samples = model.sample_with_composition(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    instruction_ids=instruction_ids,
                    instruction_mask=instruction_mask,
                    composer=composer,
                    device=device,
                )
        else:
            # Standard sampling
            if lora_module:
                instruction_emb = lora_module.instruction_encoder(
                    instruction_ids,
                    attention_mask=instruction_mask,
                )
                instruction_emb = lora_module.instruction_to_hidden(instruction_emb)
            elif instruction_encoder is not None and instruction_to_hidden is not None:
                instruction_emb = instruction_encoder(instruction_ids, attention_mask=instruction_mask)
                instruction_emb = instruction_to_hidden(instruction_emb)
            else:
                instruction_emb = None
            
            # Use classification-specific sampling if available
            if is_classification and hasattr(model, 'sample_classification'):
                samples = model.sample_classification(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    instruction_embedding=instruction_emb,
                    device=device,
                    tokenizer=tokenizer,
                    label_names=label_names,
                    early_stop=True,
                )
            else:
                # Use standard sampling with greedy option for classification
                samples = model.sample(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    instruction_embedding=instruction_emb,
                    device=device,
                    greedy=is_classification,  # Use greedy for classification
                )
        
        # Decode predictions
        for sample in samples:
            pred_text = tokenizer.decode(sample, skip_special_tokens=True)
            all_predictions.append(pred_text)
    
    return all_predictions, all_references


@torch.no_grad()
def get_final_hiddens_and_labels(
    model,
    lora_module,
    dataloader,
    device,
    config,
    label_names,
    classification_seq_len=5,
    composer=None,
    instruction_encoder=None,
    instruction_to_hidden=None,
):
    """
    Collect final-step hidden states and label indices for classification-head evaluation.
    Returns (hiddens, label_indices) where hiddens is (N, hidden_dim), label_indices (N,) 0/1.
    When composer is provided (multi-task composition), uses sample_with_composition.
    """
    model.eval()
    if lora_module:
        lora_module.eval()
    if composer:
        composer.eval()
    
    all_hiddens = []
    all_label_indices = []
    
    for batch in tqdm(dataloader, desc="Final hiddens"):
        batch_size = batch["target_ids"].size(0)
        seq_len = classification_seq_len
        instruction_ids = batch["instruction_ids"].to(device)
        instruction_mask = batch["instruction_mask"].to(device)
        target_texts = batch.get("target_texts", [])
        
        if composer is not None:
            xt, final_hidden = model.sample_with_composition(
                batch_size=batch_size,
                seq_len=seq_len,
                instruction_ids=instruction_ids,
                instruction_mask=instruction_mask,
                composer=composer,
                device=device,
                return_final_hidden=True,
            )
        elif lora_module:
            xt, final_hidden = model.sample_with_lora(
                batch_size=batch_size,
                seq_len=seq_len,
                instruction_ids=instruction_ids,
                instruction_mask=instruction_mask,
                lora_module=lora_module,
                device=device,
                greedy=True,
                return_final_hidden=True,
            )
        else:
            inst_emb = None
            if instruction_encoder is not None and instruction_to_hidden is not None:
                inst_emb = instruction_encoder(instruction_ids, attention_mask=instruction_mask)
                inst_emb = instruction_to_hidden(inst_emb)
            xt, final_hidden = model.get_final_hidden_for_classification(
                batch_size=batch_size,
                seq_len=seq_len,
                instruction_embedding=inst_emb,
                device=device,
            )
        
        if final_hidden is not None:
            all_hiddens.append(final_hidden.cpu())
        
        for text in target_texts:
            text_lower = (text or "").strip().lower()
            idx = 0
            if label_names:
                for i, name in enumerate(label_names):
                    if name.lower() in text_lower or text_lower == name.lower():
                        idx = i
                        break
            all_label_indices.append(idx)
    
    if all_hiddens:
        hiddens = torch.cat(all_hiddens, dim=0)
        label_indices = torch.tensor(all_label_indices, dtype=torch.long)
        return hiddens, label_indices
    return None, torch.tensor(all_label_indices, dtype=torch.long)


def fit_and_eval_classification_head(
    hiddens_train, labels_train, hiddens_val, labels_val, hiddens_test, labels_test,
    hidden_dim, num_classes, device="cuda", max_epochs=50,
):
    """
    Fit a linear classification head and return val and test accuracy.
    """
    head = torch.nn.Linear(hidden_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    
    train_dataset = torch.utils.data.TensorDataset(hiddens_train, labels_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    head.train()
    for _ in range(max_epochs):
        for h, y in train_loader:
            h, y = h.to(device), y.to(device)
            logits = head(h)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    head.eval()
    with torch.no_grad():
        logits_val = head(hiddens_val.to(device))
        pred_val = logits_val.argmax(dim=-1).cpu()
        acc_val = (pred_val == labels_val).float().mean().item() * 100.0
        logits_test = head(hiddens_test.to(device))
        pred_test = logits_test.argmax(dim=-1).cpu()
        acc_test = (pred_test == labels_test).float().mean().item() * 100.0
    return acc_val, acc_test


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(Path("./logs"), experiment_name="evaluation")
    
    logger.info("=" * 80)
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Split: {args.split}")
    logger.info("=" * 80)
    
    # Load checkpoint config (try multiple locations)
    checkpoint_path = Path(args.checkpoint)
    config = find_config(checkpoint_path)
    
    if config is None:
        logger.error(f"Config not found in {checkpoint_path} or any subdirectories")
        logger.error("Tried: config.json, training_summary.json, final_model/config.json, best_model/config.json, checkpoint-*/config.json")
        return
    
    # Expected training length (from experiment dir config) for step-mismatch warning
    requested_path = checkpoint_path
    expected_max_steps = config.get("training", {}).get("max_steps")

    # Find the actual checkpoint subdirectory containing model files
    actual_checkpoint_dir = find_checkpoint_subdir(checkpoint_path, config=config)
    if actual_checkpoint_dir is None:
        try:
            path_str = requested_path.resolve().as_posix()
            if "multi_seed_experiments" in path_str and "_seed" in requested_path.name:
                logger.error(
                    "No checkpoint in %s (missing final_model/model.pt or model.pt). "
                    "Re-run training for this seed so the experiment dir contains the checkpoint, "
                    "or evaluate will use a different seed's checkpoint and report wrong val accuracy.",
                    requested_path,
                )
                return
        except (OSError, RuntimeError):
            pass
        logger.warning(f"Could not find model.pt or lora_module.pt in {checkpoint_path} or subdirectories")
        logger.warning("Will try to load from checkpoint_path directly")
        actual_checkpoint_dir = checkpoint_path
    else:
        # Update checkpoint_path to point to the actual checkpoint directory
        checkpoint_path = actual_checkpoint_dir

    # Warn if we resolved to a different dir and the checkpoint step doesn't match expected (e.g. loading 50-step instead of 10k)
    if actual_checkpoint_dir != requested_path and expected_max_steps is not None:
        meta_path = checkpoint_path / "checkpoint_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                ckpt_step = meta.get("step", 0)
                if ckpt_step < expected_max_steps * 0.5:
                    logger.warning(
                        "Checkpoint step mismatch: loaded from %s (step=%s) but experiment expects max_steps=%s. "
                        "Validation accuracy may be wrong (e.g. ~50%% if loading a short run).",
                        checkpoint_path, ckpt_step, expected_max_steps,
                    )
            except (json.JSONDecodeError, TypeError):
                pass

    # Use the checkpoint directory's config for seed and method so model init matches loaded weights
    # (e.g. LoRA-Diffusion: base must be initialized with same seed as the checkpoint's training)
    config_ckpt_path = checkpoint_path / "config.json"
    if config_ckpt_path.exists():
        with open(config_ckpt_path) as f:
            config_ckpt = json.load(f)
        if "training" not in config:
            config["training"] = {}
        ckpt_seed = config_ckpt.get("training", {}).get("seed")
        if ckpt_seed is not None:
            config["training"]["seed"] = ckpt_seed
            logger.info(f"Using checkpoint seed for model init: {ckpt_seed}")
        if config_ckpt.get("method") is not None:
            config["method"] = config_ckpt.get("method", config.get("method"))
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer (use local_files_only to avoid network timeouts)
    tokenizer_name = config.get("tokenizer", {}).get("name", "bert-base-uncased")
    try:
        # Try to load from cache first
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=True,
        )
    except (OSError, ValueError) as e:
        # If not in cache, try downloading (with timeout handling)
        logger.warning(f"Tokenizer not in cache, attempting download: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as download_error:
            logger.error(f"Failed to load tokenizer: {download_error}")
            raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    logger.info(f"Loaded tokenizer: {tokenizer_name}")
    
    # Load dataset
    task_config = config.get("task", {})
    
    dataset = get_task_loader(
        task_name=args.task,
        split=args.split,
        cache_dir=config["data"]["cache_dir"],
        task_config=task_config,
        tokenizer=tokenizer,
        max_samples=None,
        max_seq_length=config["model"]["max_seq_length"],
    )
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Create dataloader
    collator = DiffusionCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )
    
    # Load model
    logger.info("Loading model...")
    
    # For lora_diffusion, the base model is frozen and not saved; only lora_module.pt exists.
    # We must use the same random seed as training so the base model initializes identically.
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    
    # Check method type from config to determine if we need to apply PEFT modules
    method_name = config.get("method", {}).get("name", None)
    
    model = MaskedDiffusionTransformer(config)
    
    # For weight_lora, adapters, and bitfit, apply PEFT modules before loading
    # because they're embedded in the model checkpoint
    if method_name == "weight_lora":
        from src.models.baselines import apply_weight_lora_to_model
        # Get config from method section or lora section
        method_config = config.get("method", {})
        lora_config = config.get("lora", {})
        # Try to get from method.lora first, then lora section
        method_lora = method_config.get("lora", {})
        # Use rank from config (default to 64 if not found)
        rank = method_lora.get("rank") or lora_config.get("rank") or lora_config.get("rank_early") or 64
        alpha = method_lora.get("alpha") or lora_config.get("alpha") or lora_config.get("scaling_early") or 16
        target_modules = method_lora.get("target_modules") or lora_config.get("target_modules") or ["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"]
        apply_weight_lora_to_model(model, rank=rank, alpha=alpha, target_modules=target_modules)
        logger.info(f"Applied weight LoRA modules to model (rank={rank}, alpha={alpha}, {len(target_modules)} target modules)")
    elif method_name == "adapters":
        from src.models.baselines import apply_adapters_to_model
        method_adapter = config.get("method", {}).get("adapter", {})
        adapter_config = {**config.get("adapter", {}), **method_adapter}
        bottleneck_dim = adapter_config.get("bottleneck_dim", 256)
        insert_after = adapter_config.get("insert_after", ["attention", "ffn"])
        apply_adapters_to_model(model, bottleneck_dim=bottleneck_dim, insert_after=insert_after)
        logger.info("Applied adapter modules to model")
    elif method_name == "bitfit":
        # BitFit just requires bias parameters to be trainable, which they already are
        # No special setup needed
        pass
    
    # Move model to device AFTER applying PEFT modules (so LoRA modules are also on device)
    model = model.to(device)
    
    # Load model weights
    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        # Use strict=False for weight_lora/adapters/bitfit since they may have extra LoRA/adapter keys
        strict_loading = method_name not in ["weight_lora", "adapters", "bitfit"]
        state_dict = torch.load(model_path, map_location=device)
        if strict_loading:
            model.load_state_dict(state_dict)
        else:
            # Load with strict=False; log missing/unexpected to verify PEFT weights load
            load_result = model.load_state_dict(state_dict, strict=False)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            if missing:
                logger.warning(f"Loaded model with strict=False: {len(missing)} missing keys (model expected but not in checkpoint)")
                if any("lora_module" in k or "adapter_module" in k for k in missing):
                    logger.warning("Some PEFT weights are missing — validation accuracy may be wrong. First few: %s", missing[:5])
                else:
                    logger.debug("Missing keys (first 5): %s", missing[:5])
            if unexpected:
                logger.warning(f"Loaded model with strict=False: {len(unexpected)} unexpected keys (in checkpoint but not in model). First few: %s", unexpected[:5])
            if not missing and not unexpected:
                logger.info("Loaded model with strict=False: all checkpoint keys matched (PEFT method)")
            else:
                logger.info("Loaded model with strict=False (PEFT method)")
    
    # Load LoRA module or composer (or shared instruction encoder for baselines)
    lora_module = None
    composer = None
    instruction_encoder = None
    instruction_to_hidden = None
    
    if args.use_composition:
        if not args.task_modules or not args.task_names:
            raise ValueError(
                "When using --use_composition, must provide --task_modules and --task_names"
            )
        if len(args.task_modules) != len(args.task_names):
            raise ValueError(
                f"Number of task_modules ({len(args.task_modules)}) must match "
                f"number of task_names ({len(args.task_names)})"
            )
        
        logger.info("Loading multi-task composer...")
        composer = MultiTaskLoRAComposer(
            config=config,
            task_names=args.task_names,
        ).to(device)
        
        # Load task-specific LoRA modules
        for task_name, task_checkpoint in zip(args.task_names, args.task_modules):
            task_checkpoint_path = Path(task_checkpoint)
            if not task_checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {task_checkpoint_path}")
            
            logger.info(f"Loading LoRA module for {task_name} from {task_checkpoint_path}")
            composer.load_task_module(task_name, task_checkpoint_path)
        
        # Load router if available
        router_path = Path(args.router_path) if args.router_path else Path(args.task_modules[0]) / "router.pt"
        if router_path.exists():
            logger.info(f"Loading router from {router_path}")
            router_state = torch.load(router_path, map_location=device)
            composer.router.load_state_dict(router_state)
        composer.composition_mode = args.composition_mode
        logger.info(f"Composition mode: {composer.composition_mode}")
    else:
        # Load single LoRA module
        lora_path = checkpoint_path / "lora_module.pt"
        if lora_path.exists():
            logger.info("Loading LoRA module...")
            lora_module = LoRADiffusionModule(config).to(device)
            lora_module.load_state_dict(torch.load(lora_path, map_location=device))
        # Load shared instruction encoder for baselines (full_ft, weight_lora, adapters, bitfit)
        enc_path = checkpoint_path / "instruction_encoder.pt"
        if enc_path.exists() and lora_module is None:
            logger.info("Loading shared instruction encoder (baseline)...")
            lora_cfg = config.get("lora", {})
            model_cfg = config.get("model", {})
            instruction_encoder = InstructionEncoder(
                vocab_size=model_cfg.get("vocab_size", 30522),
                hidden_dim=model_cfg.get("hidden_dim", 768),
                output_dim=lora_cfg.get("instruction_encoder_hidden", 256),
                num_layers=lora_cfg.get("instruction_encoder_layers", 2),
            ).to(device)
            instruction_to_hidden = torch.nn.Linear(
                lora_cfg.get("instruction_encoder_hidden", 256),
                model_cfg.get("hidden_dim", 768),
            ).to(device)
            enc_data = torch.load(enc_path, map_location=device)
            instruction_encoder.load_state_dict(enc_data["instruction_encoder"])
            instruction_to_hidden.load_state_dict(enc_data["instruction_to_hidden"])
    
    # Classification-head evaluation (primary classification metric)
    metrics_extra = {}
    if args.eval_classification_head:
        task_type = task_config.get("type", "classification")
        label_names = task_config.get("label_names", ["negative", "positive"])
        num_classes = task_config.get("num_labels", len(label_names))
        hidden_dim = config["model"]["hidden_dim"]
        collator = DiffusionCollator(tokenizer=tokenizer)
        
        def make_loader(split_name):
            ds = get_task_loader(
                task_name=args.task,
                split=split_name,
                cache_dir=config["data"]["cache_dir"],
                task_config=task_config,
                tokenizer=tokenizer,
                max_samples=None,
                max_seq_length=config["model"]["max_seq_length"],
            )
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=2)
        
        train_loader = make_loader("train")
        val_loader = make_loader("validation")
        try:
            test_loader = make_loader("test")
        except (ValueError, KeyError):
            test_loader = make_loader("validation")
        
        logger.info("Collecting final hiddens for classification head (train)...")
        h_train, y_train = get_final_hiddens_and_labels(
            model, lora_module, train_loader, device, config, label_names,
            composer=composer,
            instruction_encoder=instruction_encoder,
            instruction_to_hidden=instruction_to_hidden,
        )
        logger.info("Collecting final hiddens (validation)...")
        h_val, y_val = get_final_hiddens_and_labels(
            model, lora_module, val_loader, device, config, label_names,
            composer=composer,
            instruction_encoder=instruction_encoder,
            instruction_to_hidden=instruction_to_hidden,
        )
        logger.info("Collecting final hiddens (test)...")
        h_test, y_test = get_final_hiddens_and_labels(
            model, lora_module, test_loader, device, config, label_names,
            composer=composer,
            instruction_encoder=instruction_encoder,
            instruction_to_hidden=instruction_to_hidden,
        )
        
        if h_train is not None and h_val is not None and h_test is not None:
            acc_val, acc_test = fit_and_eval_classification_head(
                h_train, y_train, h_val, y_val, h_test, y_test,
                hidden_dim, num_classes, device=device,
            )
            logger.info(f"Classification-head Val acc.: {acc_val:.2f}%")
            logger.info(f"Classification-head Test acc.: {acc_test:.2f}%")
            results_ch = {"classification_head_val_acc": acc_val, "classification_head_test_acc": acc_test}
            if args.output_file:
                ch_path = Path(args.output_file).with_suffix(".classification_head.json")
                with open(ch_path, "w") as f:
                    json.dump(results_ch, f, indent=2)
                logger.info(f"Saved classification-head results to {ch_path}")
        else:
            logger.warning("Final hiddens not collected; skipping classification-head eval.")
    
    task_type = task_config.get("type", "classification")
    use_teacher_forced = args.teacher_forced and task_type == "classification"
    
    # Ensure eval mode for teacher-forced and generation (dropout off, etc.)
    model.eval()
    if lora_module is not None:
        lora_module.eval()
    if composer is not None:
        composer.eval()
    
    if use_teacher_forced:
        # Primary metric: teacher-forced token-level accuracy (same metric as training)
        logger.info("Running teacher-forced evaluation (token-level, same metric as training)...")
        metrics = compute_teacher_forced_metrics(
            model=model,
            lora_module=lora_module,
            dataloader=dataloader,
            device=device,
            config=config,
            instruction_encoder=instruction_encoder,
            instruction_to_hidden=instruction_to_hidden,
        )
        predictions, references = None, None
        if args.run_generation:
            logger.info("Generating predictions (saving generation_accuracy)...")
            predictions, references = generate_predictions(
                model=model,
                lora_module=lora_module,
                composer=composer,
                dataloader=dataloader,
                device=device,
                config=config,
                use_composition=args.use_composition,
                instruction_encoder=instruction_encoder,
                instruction_to_hidden=instruction_to_hidden,
            )
            tokenizer = dataloader.dataset.tokenizer if hasattr(dataloader.dataset, 'tokenizer') else None
        gen_metrics = compute_metrics(
            predictions=predictions,
            references=references,
            task_config=config,
            tokenizer=tokenizer,
        )
        metrics["generation_accuracy"] = gen_metrics.get("accuracy")
    else:
        # Generation-based evaluation (for QA, summarization, or when --no_teacher_forced)
        # Do NOT save generation under "accuracy" so collector/paper never use it as val_accuracy.
        logger.info("Generating predictions...")
        predictions, references = generate_predictions(
            model=model,
            lora_module=lora_module,
            composer=composer,
            dataloader=dataloader,
            device=device,
            config=config,
            use_composition=args.use_composition,
            instruction_encoder=instruction_encoder,
            instruction_to_hidden=instruction_to_hidden,
        )
        tokenizer = dataloader.dataset.tokenizer if hasattr(dataloader.dataset, 'tokenizer') else None
        gen_metrics = compute_metrics(
            predictions=predictions,
            references=references,
            task_config=config,
            tokenizer=tokenizer,
        )
        metrics = {"generation_accuracy": gen_metrics.get("accuracy"), "loss": None}
        for k, v in gen_metrics.items():
            if k != "accuracy":
                metrics[k] = v
    
    metrics.update(metrics_extra)
    
    # Log results
    logger.info("Results:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")
    
    # Save results
    results = {
        "checkpoint": str(checkpoint_path),
        "task": args.task,
        "split": args.split,
        "num_samples": len(predictions) if predictions else len(dataloader.dataset),
        "metrics": metrics,
    }
    
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")
        # Structured metrics.json for aggregation (token vs GLUE, no mixing)
        m = results.get("metrics", {})
        structured = {
            "token_accuracy_val": m.get("accuracy"),
            "token_accuracy_test": m.get("accuracy") if args.split == "test" else None,
            "glue_accuracy_val": m.get("generation_accuracy"),
            "glue_accuracy_test": m.get("generation_accuracy") if args.split == "test" else None,
            "glue_f1_val": m.get("f1"),
            "glue_f1_test": m.get("f1") if args.split == "test" else None,
            "loss": m.get("loss"),
        }
        metrics_path = output_path.parent / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(structured, f, indent=2)
        logger.info(f"Saved structured metrics to {metrics_path}")
    
    # Optionally save predictions (only if we have them)
    if predictions:
        predictions_file = checkpoint_path / f"predictions_{args.split}.json"
        with open(predictions_file, "w") as f:
            json.dump({
                "predictions": predictions,
                "references": references,
            }, f, indent=2)
        logger.info(f"Saved predictions to {predictions_file}")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
