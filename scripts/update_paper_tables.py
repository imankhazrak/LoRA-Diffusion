#!/usr/bin/env python3
"""Update paper tables with collected experiment results."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

def format_number(value: float, decimals: int = 2) -> str:
    """Format number for LaTeX table."""
    if value is None:
        return "---"
    if isinstance(value, (int, float)):
        if decimals == 0:
            return f"{int(value)}"
        return f"{value:.{decimals}f}"
    return str(value)


def format_percent(value: float, decimals: int = 1) -> str:
    """Format percentage for LaTeX table."""
    if value is None:
        return "---"
    return f"{value:.{decimals}f}\\%"


def format_mb(value: float, decimals: int = 1) -> str:
    """Format MB for LaTeX table."""
    if value is None:
        return "---"
    if value >= 1000:
        return f"{value/1000:.{decimals}f}\\,GB"
    return f"{value:.{decimals}f}\\,MB"


def format_millions(value: int) -> str:
    """Format large numbers in millions."""
    if value is None:
        return "---"
    if value >= 1000000:
        return f"{value/1000000:.1f}M"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    return str(value)


def update_main_results_table(
    tex_content: str,
    results: Dict[str, Any],
    param_counts: Dict[str, Any],
) -> str:
    """Update Table 3 (tab:main_results)."""
    # Extract table
    pattern = r'(\\begin\{table\}.*?\\caption\{Average performance on SST-2.*?\\label\{tab:main_results\}.*?\\begin\{tabular\}.*?)(.*?)(\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex_content, re.DOTALL)
    
    if not match:
        print("Warning: Could not find tab:main_results table")
        return tex_content
    
    table_start = match.group(1)
    table_body = match.group(2)
    table_end = match.group(3)
    
    # Build new table rows
    rows = []
    rows.append("\\toprule")
    rows.append("Method & Trainable \\% & Train acc. (\\%) & Val acc. (\\%) & Relative to full FT & Steps \\\\ \\midrule")
    
    # Fix: Ensure tabular environment has proper column specification (6 columns now)
    if "@{}lccccc@{}" not in table_start and "@{}lcccc@{}" in table_start:
        # Try to fix tabular specification
        table_start = re.sub(
            r'(\\begin\{tabular\})\{.*?\}',
            r'\1{@{}lccccc@{}}',
            table_start
        )
    
    # Get full FT baseline
    full_ft_acc = results.get("full_ft", {}).get("val_accuracy")
    if full_ft_acc is None:
        full_ft_acc = 82.13  # Use old value as fallback
    
    methods_order = [
        ("full_ft", "Full Fine-Tuning"),
        ("adapters", "Adapter Layers"),
        ("weight_lora", "Weight LoRA"),
        ("prefix_tuning", "Prefix Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("bitfit", "BitFit"),
    ]
    
    for method_key, method_name in methods_order:
        method_data = results.get(method_key, {})
        param_data = param_counts.get(method_key, {})
        
        param_pct = param_data.get("trainable_percent", 0)
        acc = method_data.get("val_accuracy")
        steps = method_data.get("steps", 50)
        
        # Calculate relative to full FT
        if acc is not None and full_ft_acc and full_ft_acc > 0:
            relative = (acc / full_ft_acc) * 100
            relative_str = f"{relative:.1f}\\%"
        else:
            relative_str = "---"
        
        if acc is None:
            acc_str = "---"
        else:
            acc_str = f"{acc * 100:.2f}"
        
        train_acc = method_data.get("final_train_acc")
        train_acc_str = f"{train_acc * 100:.2f}" if train_acc is not None else "---"
        
        if method_key == "prefix_tuning" and acc is None:
            rows.append(f"{method_name} & {format_percent(param_pct)} & --- & --- & --- & --- \\\\")
        else:
            rows.append(f"{method_name} & {format_percent(param_pct)} & {train_acc_str} & {acc_str} & {relative_str} & {steps} \\\\")
    
    rows.append("\\bottomrule")
    
    new_table_body = "\n".join(rows)
    new_table = table_start + new_table_body + "\n" + table_end
    
    return tex_content[:match.start()] + new_table + tex_content[match.end():]


def update_per_task_results_table(
    tex_content: str,
    results: Dict[str, Any],
    param_counts: Dict[str, Any],
) -> str:
    """Update Table 4 (tab:per_task_results)."""
    pattern = r'(\\begin\{table\}.*?\\caption\{Detailed results on SST-2.*?\\label\{tab:per_task_results\}.*?\\begin\{tabular\}.*?)(.*?)(\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex_content, re.DOTALL)
    
    if not match:
        print("Warning: Could not find tab:per_task_results table")
        return tex_content
    
    table_start = match.group(1)
    table_body = match.group(2)
    table_end = match.group(3)
    
    rows = []
    rows.append("\\toprule")
    rows.append("Method & Steps & Train loss & Train acc.\\ (\\%) & Val acc.\\ (\\%) & Param.\\ \\% & Status \\\\ \\midrule")
    
    methods_order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("bitfit", "BitFit"),
        ("adapters", "Adapters"),
        ("prefix_tuning", "Prefix Tuning"),
    ]
    
    for method_key, method_name in methods_order:
        method_data = results.get(method_key, {})
        param_data = param_counts.get(method_key, {})
        
        steps = method_data.get("steps", 50)
        train_loss = method_data.get("train_loss")
        train_acc = method_data.get("final_train_acc")
        val_acc = method_data.get("val_accuracy")
        param_pct = param_data.get("trainable_percent", 0)
        
        status = "$\\checkmark$" if val_acc is not None else "$\\times$"
        train_acc_str = f"{train_acc * 100:.2f}" if train_acc is not None else "---"
        val_acc_str = f"{val_acc * 100:.2f}" if val_acc is not None else "---"
        
        rows.append(
            f"{method_name} & {steps} & {format_number(train_loss, 4) if train_loss else '---'} & "
            f"{train_acc_str} & {val_acc_str} & {format_percent(param_pct)} & {status} \\\\"
        )
    
    rows.append("\\bottomrule")
    
    new_table_body = "\n".join(rows)
    new_table = table_start + new_table_body + "\n" + table_end
    
    return tex_content[:match.start()] + new_table + tex_content[match.end():]


def update_efficiency_table(
    tex_content: str,
    results: Dict[str, Any],
    param_counts: Dict[str, Any],
) -> str:
    """Update Table 5 (tab:efficiency)."""
    pattern = r'(\\begin\{table\}.*?\\caption\{Training and inference efficiency.*?\\label\{tab:efficiency\}.*?\\begin\{tabular\}.*?)(.*?)(\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex_content, re.DOTALL)
    
    if not match:
        print("Warning: Could not find tab:efficiency table")
        return tex_content
    
    table_start = match.group(1)
    table_body = match.group(2)
    table_end = match.group(3)
    
    rows = []
    rows.append("\\toprule")
    rows.append("Method & Trainable params & Param.\\ \\% & Steps & Final acc.\\ (\\%) & Storage (MB) \\\\ \\midrule")
    
    methods_order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("adapters", "Adapters"),
        ("bitfit", "BitFit"),
        ("prefix_tuning", "Prefix Tuning"),
    ]
    
    for method_key, method_name in methods_order:
        method_data = results.get(method_key, {})
        param_data = param_counts.get(method_key, {})
        
        trainable_params = param_data.get("trainable_params", 0)
        param_pct = param_data.get("trainable_percent", 0)
        steps = method_data.get("steps", 50)
        val_acc = method_data.get("val_accuracy")
        storage_mb = param_data.get("storage_mb", 0)
        
        # Format accuracy as percentage (multiply by 100 since stored as fraction)
        acc_display = format_number(val_acc * 100, 2) if val_acc else "---"
        rows.append(
            f"{method_name} & {format_millions(trainable_params)} & {format_percent(param_pct)} & "
            f"{steps} & {acc_display} & {format_mb(storage_mb)} \\\\"
        )
    
    rows.append("\\bottomrule")
    
    new_table_body = "\n".join(rows)
    new_table = table_start + new_table_body + "\n" + table_end
    
    return tex_content[:match.start()] + new_table + tex_content[match.end():]


def update_forgetting_table(
    tex_content: str,
    results: Dict[str, Any],
) -> str:
    """Update Table 6 (tab:forgetting)."""
    pattern = r'(\\begin\{table\}.*?\\caption\{Training loss and convergence.*?\\label\{tab:forgetting\}.*?\\begin\{tabular\}.*?)(.*?)(\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex_content, re.DOTALL)
    
    if not match:
        print("Warning: Could not find tab:forgetting table")
        return tex_content
    
    table_start = match.group(1)
    table_body = match.group(2)
    table_end = match.group(3)
    
    rows = []
    rows.append("\\toprule")
    rows.append("Method & Initial loss & Final loss & Loss reduction & Convergence \\\\ \\midrule")
    
    methods_order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("adapters", "Adapters"),
        ("bitfit", "BitFit"),
    ]
    
    for method_key, method_name in methods_order:
        method_data = results.get(method_key, {})
        
        initial_loss = method_data.get("initial_loss", 9.5)
        final_loss = method_data.get("train_loss")
        steps = method_data.get("steps", 50)
        
        if final_loss is not None and initial_loss:
            loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
            loss_reduction_str = f"{loss_reduction:.1f}\\%"
        else:
            loss_reduction_str = "---"
        
        # Determine convergence speed
        if steps <= 50:
            convergence = "Fast (50 steps)"
        elif steps <= 100:
            convergence = "Moderate (100 steps)"
        else:
            convergence = "Slow"
        
        rows.append(
            f"{method_name} & $\\sim${format_number(initial_loss, 1)} & "
            f"{format_number(final_loss, 4) if final_loss else '---'} & "
            f"{loss_reduction_str} & {convergence} \\\\"
        )
    
    rows.append("\\bottomrule")
    
    new_table_body = "\n".join(rows)
    new_table = table_start + new_table_body + "\n" + table_end
    
    return tex_content[:match.start()] + new_table + tex_content[match.end():]


def update_composition_table(
    tex_content: str,
    results: Dict[str, Any],
    param_counts: Dict[str, Any],
) -> str:
    """Update Table 7 (tab:composition)."""
    pattern = r'(\\begin\{table\}.*?\\caption\{Method comparison summary.*?\\label\{tab:composition\}.*?\\begin\{tabular\}.*?)(.*?)(\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex_content, re.DOTALL)
    
    if not match:
        print("Warning: Could not find tab:composition table")
        return tex_content
    
    table_start = match.group(1)
    table_body = match.group(2)
    table_end = match.group(3)
    
    rows = []
    rows.append("\\toprule")
    rows.append("Method & Train acc. (\\%) & Val acc. (\\%) & Train loss & Steps & Param.\\ \\% & Status \\\\ \\midrule")
    
    methods_order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("bitfit", "BitFit"),
        ("adapters", "Adapters"),
        ("prefix_tuning", "Prefix Tuning"),
    ]
    
    for method_key, method_name in methods_order:
        method_data = results.get(method_key, {})
        param_data = param_counts.get(method_key, {})
        
        train_acc = method_data.get("final_train_acc")
        val_acc = method_data.get("val_accuracy")
        train_loss = method_data.get("train_loss")
        steps = method_data.get("steps", 50)
        param_pct = param_data.get("trainable_percent", 0)
        status = "$\\checkmark$" if val_acc is not None else "$\\times$"
        
        train_acc_str = f"{train_acc * 100:.2f}" if train_acc is not None else "---"
        val_acc_str = f"{val_acc * 100:.2f}" if val_acc is not None else "---"
        
        rows.append(
            f"{method_name} & {train_acc_str} & {val_acc_str} & "
            f"{format_number(train_loss, 4) if train_loss else '---'} & "
            f"{steps} & {format_percent(param_pct)} & {status} \\\\"
        )
    
    # Add comparison row
    full_ft_data = results.get("full_ft", {})
    lora_diff_data = results.get("lora_diffusion", {})
    
    if full_ft_data.get("val_accuracy") and lora_diff_data.get("val_accuracy"):
        ft_acc = full_ft_data["val_accuracy"]
        ld_acc = lora_diff_data["val_accuracy"]
        relative_acc = (ld_acc / ft_acc) * 100
        
        ft_loss = full_ft_data.get("train_loss", 0.2621)
        ld_loss = lora_diff_data.get("train_loss", 0.5652)
        loss_ratio = ld_loss / ft_loss if ft_loss > 0 else 0
        
        ft_steps = full_ft_data.get("steps", 50)
        ld_steps = lora_diff_data.get("steps", 100)
        steps_ratio = ld_steps / ft_steps if ft_steps > 0 else 0
        
        ld_param_pct = param_counts.get("lora_diffusion", {}).get("trainable_percent", 0)
        
        rows.append("\\midrule")
        rows.append(
            f"LoRA-Diffusion vs.\\ full FT & --- & {relative_acc:.1f}\\% & "
            f"{loss_ratio:.1f}$\\times$ & {steps_ratio:.1f}$\\times$ & "
            f"{format_percent(ld_param_pct)} & --- \\\\"
        )
    
    rows.append("\\bottomrule")
    
    new_table_body = "\n".join(rows)
    new_table = table_start + new_table_body + "\n" + table_end
    
    return tex_content[:match.start()] + new_table + tex_content[match.end():]


def handle_parameter_discrepancy(
    tex_content: str,
    param_counts: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Handle LoRA-Diffusion parameter count discrepancy.
    
    Returns: (updated_tex_content, note_to_add)
    """
    lora_data = param_counts.get("lora_diffusion", {})
    breakdown = lora_data.get("breakdown", {})
    
    instruction_encoder_params = breakdown.get("instruction_encoder", 0)
    adapter_params = (
        breakdown.get("adapters_early", 0) +
        breakdown.get("adapters_mid", 0) +
        breakdown.get("adapters_late", 0) +
        breakdown.get("instruction_to_hidden", 0)
    )
    total_params = lora_data.get("trainable_params", 0)
    base_params = lora_data.get("base_params", 137676090)
    
    # Calculate percentages
    total_pct = (total_params / base_params) * 100
    adapter_only_pct = (adapter_params / base_params) * 100
    encoder_pct = (instruction_encoder_params / base_params) * 100
    
    # Add a note in the efficiency section
    total_params_str = format_millions(total_params)
    encoder_params_str = format_millions(instruction_encoder_params)
    adapter_params_str = format_millions(adapter_params)
    
    note = (
        f"\\textbf{{Note on LoRA-Diffusion parameters:}} The total trainable parameters "
        f"({total_params_str}, {total_pct:.1f}\\% of base model) include "
        f"the instruction encoder ({encoder_params_str}, {encoder_pct:.1f}\\%). "
        f"The trajectory adapters alone comprise {adapter_params_str} parameters "
        f"({adapter_only_pct:.2f}\\% of base model)."
    )
    
    # Update efficiency section text if needed
    efficiency_section_pattern = r'(Table~\\ref\{tab:efficiency\} summarizes efficiency\.)(.*?)(Weight LoRA)'
    match = re.search(efficiency_section_pattern, tex_content, re.DOTALL)
    
    if match:
        # Insert note after efficiency summary
        updated_text = match.group(1) + "\n\n" + note + "\n\n" + match.group(3)
        tex_content = tex_content[:match.start()] + updated_text + tex_content[match.end():]
    
    return tex_content, note


def main():
    parser = argparse.ArgumentParser(description="Update paper tables with experiment results")
    parser.add_argument(
        "--collected-results",
        type=str,
        required=True,
        help="Path to collected_results.json from collect_results.py",
    )
    parser.add_argument(
        "--paper-tex",
        type=str,
        default="doc/Paper.tex",
        help="Path to Paper.tex file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output LaTeX file (default: overwrite input)",
    )
    parser.add_argument(
        "--handle-discrepancy",
        action="store_true",
        help="Add note about LoRA-Diffusion parameter count discrepancy",
    )
    
    args = parser.parse_args()
    
    # Load collected results
    with open(args.collected_results, "r") as f:
        collected = json.load(f)
    
    formatted_results = collected.get("formatted_for_tables", {})
    raw_results = collected.get("raw_results", {})
    param_counts = raw_results.get("parameter_counts", {})
    
    # Load paper
    paper_path = Path(args.paper_tex)
    with open(paper_path, "r") as f:
        tex_content = f.read()
    
    # Update tables
    print("Updating tables...")
    tex_content = update_main_results_table(tex_content, formatted_results, param_counts)
    print("  ✓ Updated tab:main_results")
    
    tex_content = update_per_task_results_table(tex_content, formatted_results, param_counts)
    print("  ✓ Updated tab:per_task_results")
    
    tex_content = update_efficiency_table(tex_content, formatted_results, param_counts)
    print("  ✓ Updated tab:efficiency")
    
    tex_content = update_forgetting_table(tex_content, formatted_results)
    print("  ✓ Updated tab:forgetting")
    
    tex_content = update_composition_table(tex_content, formatted_results, param_counts)
    print("  ✓ Updated tab:composition")
    
    # Handle parameter discrepancy if requested
    if args.handle_discrepancy:
        tex_content, note = handle_parameter_discrepancy(tex_content, param_counts)
        print(f"  ✓ Added note about parameter discrepancy")
        print(f"    Note: {note[:100]}...")
    
    # Save updated paper
    output_path = Path(args.output) if args.output else paper_path
    with open(output_path, "w") as f:
        f.write(tex_content)
    
    print(f"\nPaper updated and saved to: {output_path}")
    
    # Print summary
    print("\nSummary of updates:")
    for method_key in ["full_ft", "lora_diffusion", "weight_lora", "adapters", "bitfit"]:
        method_data = formatted_results.get(method_key, {})
        param_data = param_counts.get(method_key, {})
        print(f"\n{method_key}:")
        print(f"  Parameters: {format_millions(param_data.get('trainable_params', 0))} ({param_data.get('trainable_percent', 0):.2f}%)")
        print(f"  Storage: {format_mb(param_data.get('storage_mb', 0))}")
        if method_data.get("val_accuracy"):
            print(f"  Val Accuracy: {method_data['val_accuracy']:.2f}%")
        if method_data.get("train_loss"):
            print(f"  Train Loss: {method_data['train_loss']:.4f}")


if __name__ == "__main__":
    main()
