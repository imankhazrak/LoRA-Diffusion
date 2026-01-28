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
from src.data import get_task_loader, DiffusionCollator
from src.evaluation import compute_metrics
from src.utils import load_config, setup_logging

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
    
    return parser.parse_args()


@torch.no_grad()
def generate_predictions(
    model, 
    lora_module, 
    composer, 
    dataloader, 
    device, 
    config,
    use_composition=False,
):
    """Generate predictions for all samples."""
    model.eval()
    if lora_module:
        lora_module.eval()
    if composer:
        composer.eval()
    
    all_predictions = []
    all_references = []
    
    for batch in tqdm(dataloader, desc="Generating"):
        target_texts = batch.get("target_texts", [])
        all_references.extend(target_texts)
        
        # For simplicity, we'll use greedy decoding from the model
        # In practice, you'd want beam search or other decoding strategies
        
        batch_size = batch["target_ids"].size(0)
        seq_len = batch["target_ids"].size(1)
        
        instruction_ids = batch["instruction_ids"].to(device)
        instruction_mask = batch["instruction_mask"].to(device)
        
        if use_composition and composer:
            # Use composition sampling
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
            else:
                instruction_emb = None
            
            samples = model.sample(
                batch_size=batch_size,
                seq_len=seq_len,
                instruction_embedding=instruction_emb,
                device=device,
            )
        
        # Decode predictions
        tokenizer = dataloader.dataset.tokenizer
        for sample in samples:
            pred_text = tokenizer.decode(sample, skip_special_tokens=True)
            all_predictions.append(pred_text)
    
    return all_predictions, all_references


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
    
    # Load checkpoint config
    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path / "config.json"
    
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    model = MaskedDiffusionTransformer(config).to(device)
    
    # Load model weights
    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load LoRA module or composer
    lora_module = None
    composer = None
    
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
        router_path = Path(args.task_modules[0]) / "router.pt"
        if router_path.exists():
            logger.info(f"Loading router from {router_path}")
            router_state = torch.load(router_path, map_location=device)
            composer.router.load_state_dict(router_state)
    else:
        # Load single LoRA module
        lora_path = checkpoint_path / "lora_module.pt"
        if lora_path.exists():
            logger.info("Loading LoRA module...")
            lora_module = LoRADiffusionModule(config).to(device)
            lora_module.load_state_dict(torch.load(lora_path, map_location=device))
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions, references = generate_predictions(
        model=model,
        lora_module=lora_module,
        composer=composer,
        dataloader=dataloader,
        device=device,
        config=config,
        use_composition=args.use_composition,
    )
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(
        predictions=predictions,
        references=references,
        task_config=task_config,
    )
    
    # Log results
    logger.info("Results:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Save results
    results = {
        "checkpoint": str(checkpoint_path),
        "task": args.task,
        "split": args.split,
        "num_samples": len(predictions),
        "metrics": metrics,
    }
    
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")
    
    # Optionally save predictions
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
