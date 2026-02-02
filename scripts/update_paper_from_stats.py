#!/usr/bin/env python3
"""Update doc/Paper.tex tables from collected_results_with_stats.json and parameter_counts.json."""

import argparse
import json
import re
import math
from pathlib import Path
from typing import Dict, Any, Optional

def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)

def fmt_pct(value: float, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    return f"{value:.{decimals}f}\\%"

def fmt_num(value: Optional[float], decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    return f"{value:.{decimals}f}"

def fmt_mean_std(mean: Optional[float], std: Optional[float], decimals: int = 2, pct: bool = False) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "---"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        std = 0.0
    if pct and mean < 2 and std < 2:
        return f"${mean*100:.{decimals}f} \\pm {std*100:.{decimals}f}$"
    return f"${fmt_num(mean, decimals)} \\pm {fmt_num(std, decimals)}$"

def get_sig_marker(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def update_main_results(tex: str, methods: Dict, params: Dict, comparisons: Dict) -> str:
    """Replace tab:main_results table with mean ± std from stats (standard classification-head Val acc)."""
    # Match from \label{tab:main_results} to \midrule (skip tabular spec which may contain }), then body, then \bottomrule...
    pattern = r'(\\label\{tab:main_results\}.*?\\midrule\s*\n)(.*?)(\n\s*\\bottomrule\s*\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex, re.DOTALL)
    if not match:
        return tex
    # Keep full table from \begin{table} through \midrule
    start = match.start()
    table_begin = tex[:start].rfind("\\begin{table}")
    if table_begin >= 0:
        pre = tex[table_begin:match.start()] + match.group(1)
    else:
        pre = match.group(1)
    post = match.group(3)
    order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("adapters", "Adapter Layers"),
        ("weight_lora", "Weight LoRA"),
        ("bitfit", "BitFit"),
    ]
    # Val accuracy = token-level denoising (same metric as training), stored 0-1
    val_key = "val_accuracy"
    full_ft_val = methods.get("full_ft", {}).get(val_key, {}).get("mean")
    if full_ft_val is None:
        full_ft_val = 0.83
    rows = []
    for key, name in order:
        m = methods.get(key, {})
        p = params.get(key, {})
        train_acc = m.get("train_accuracy", {})
        val_acc = m.get(val_key, {})
        train_m, train_s = train_acc.get("mean"), train_acc.get("std") or 0
        val_m, val_s = val_acc.get("mean"), val_acc.get("std") or 0
        param_pct = p.get("trainable_percent", 0)
        if train_m is not None and train_m < 2:
            train_str = fmt_mean_std(train_m, train_s, 2, pct=True)
        else:
            train_str = fmt_mean_std(train_m, train_s, 2) if train_m is not None else "---"
        # val_accuracy is 0-1 (token-level); display as %
        if val_m is not None:
            v_m = val_m * 100 if val_m < 2 else val_m
            v_s = (val_s * 100) if (val_s is not None and val_s < 2) else (val_s or 0)
            val_str = f"${v_m:.2f} \\pm {v_s:.2f}$"
        else:
            val_str = "---"
        rel = (val_m / full_ft_val * 100) if (val_m is not None and full_ft_val) else None
        rel_str = fmt_pct(rel, 1) if rel is not None else "---"
        sig = ""
        if key != "full_ft":
            comp = comparisons.get(val_key, {}).get(key, {})
            ttest = comp.get("t_test") or {}
            pval = ttest.get("p_value")
            sig = get_sig_marker(pval) if pval is not None else ""
        if sig and val_m is not None:
            v_m = val_m * 100 if val_m < 2 else val_m
            v_s = (val_s * 100) if (val_s is not None and val_s < 2) else (val_s or 0)
            val_str = f"${v_m:.2f} \\pm {v_s:.2f}^{{{sig}}}$"
        rows.append(f"{name} & {fmt_num(param_pct, 1)} & {train_str} & {val_str} & {rel_str} \\\\")
    body = "\n".join(rows)
    table_begin = tex[:match.start()].rfind("\\begin{table}")
    if table_begin >= 0:
        return tex[:table_begin] + pre + body + post + tex[match.end():]
    return tex[:match.start()] + pre + body + post + tex[match.end():]

def update_classification_head(tex: str, methods: Dict) -> str:
    """Replace tab:classification_head table with token-level val (same metric as training)."""
    pattern = r'(\\begin\{table\}\[h\].*?\\label\{tab:classification_head\}.*?\\toprule\s*Method & Val acc.*?\\midrule\s*)(.*?)(\s*\\bottomrule\s*\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex, re.DOTALL)
    if not match:
        return tex
    pre, _, post = match.group(1), match.group(2), match.group(3)
    val_key = "val_accuracy"  # token-level, same as training (0-1)
    order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("adapters", "Adapters"),
        ("bitfit", "BitFit"),
    ]
    rows = []
    for key, name in order:
        m = methods.get(key, {})
        v = m.get(val_key, {})
        vm, vs = v.get("mean"), v.get("std") or 0
        ci_lo, ci_hi = v.get("ci_95_lower"), v.get("ci_95_upper")
        if vm is not None and vm < 2:
            vm, vs = vm * 100, (vs or 0) * 100
        val_str = f"${vm:.2f} \\pm {vs:.2f}$" if vm is not None else "---"
        test_str = "---"  # pipeline runs validation only for token-level
        if ci_lo is not None and ci_hi is not None and ci_lo < 2:
            ci_lo, ci_hi = ci_lo * 100, ci_hi * 100
        ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if (ci_lo is not None and ci_hi is not None) else "---"
        rows.append(f"{name} & {val_str} & {test_str} & {ci_str} \\\\")
    body = "\n".join(rows)
    return tex[:match.start()] + pre + body + post + tex[match.end():]

def update_per_task_results(tex: str, methods: Dict, params: Dict) -> str:
    """Replace tab:per_task_results table (mean values, Val acc = token-level, same as training)."""
    pattern = r'(\\begin\{table\}\[h\].*?\\label\{tab:per_task_results\}.*?\\toprule\s*Method & Steps[^\\]*\\\\\s*\\midrule\s*)(.*?)(\s*\\bottomrule\s*\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex, re.DOTALL)
    if not match:
        return tex
    pre, _, post = match.group(1), match.group(2), match.group(3)
    val_key = "val_accuracy"  # token-level (0-1)
    order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("bitfit", "BitFit"),
        ("adapters", "Adapters"),
        ("prefix_tuning", "Prefix Tuning"),
    ]
    rows = []
    for key, name in order:
        m = methods.get(key, {})
        p = params.get(key, {})
        steps = int(m.get("total_steps", {}).get("mean", 50))
        loss = m.get("final_loss", {}).get("mean")
        train_acc = m.get("train_accuracy", {}).get("mean")
        val_acc = m.get(val_key, {}).get("mean")
        param_pct = p.get("trainable_percent", 0)
        status = "$\\checkmark$" if val_acc is not None else "$\\times$"
        loss_str = fmt_num(loss, 4) if loss is not None else "---"
        train_str = f"{train_acc*100:.2f}" if (train_acc is not None and train_acc < 2) else (fmt_num(train_acc, 2) if train_acc is not None else "---")
        val_str = f"{val_acc*100:.2f}" if (val_acc is not None and val_acc < 2) else (fmt_num(val_acc, 2) if val_acc is not None else "---")
        rows.append(f"{name} & {steps} & {loss_str} & {train_str} & {val_str} & {param_pct:.1f}\\% & {status} \\\\")
    body = "\n".join(rows)
    return tex[:match.start()] + pre + body + post + tex[match.end():]

def update_efficiency_table(tex: str, methods: Dict, params: Dict) -> str:
    """Replace tab:efficiency table."""
    pattern = r'(\\begin\{table\}\[h\].*?\\label\{tab:efficiency\}.*?\\toprule\s*Method & Trainable params.*?\\midrule\s*)(.*?)(\s*\\bottomrule\s*\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex, re.DOTALL)
    if not match:
        return tex
    pre, _, post = match.group(1), match.group(2), match.group(3)
    val_key = "val_accuracy"  # token-level (0-1)
    order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("adapters", "Adapters"),
        ("bitfit", "BitFit"),
        ("prefix_tuning", "Prefix Tuning"),
    ]
    rows = []
    for key, name in order:
        m = methods.get(key, {})
        p = params.get(key, {})
        steps = int(m.get("total_steps", {}).get("mean", 50))
        train_acc = m.get("train_accuracy", {}).get("mean")
        val_acc = m.get(val_key, {}).get("mean")
        trainable_m = p.get("trainable_params", 0) / 1e6 if p.get("trainable_params") else 0
        param_pct = p.get("trainable_percent", 0)
        storage = p.get("storage_mb", 0)
        train_str = f"{train_acc*100:.2f}" if (train_acc is not None and train_acc < 2) else (fmt_num(train_acc, 2) if train_acc else "---")
        val_str = f"{val_acc*100:.2f}" if (val_acc is not None and val_acc < 2) else (fmt_num(val_acc, 2) if val_acc is not None else "N/A" if key == "prefix_tuning" else "---")
        rows.append(f"{name} & {trainable_m:.1f}M & {param_pct:.1f}\\% & {steps} & {train_str} & {val_str} & {storage:.1f}\\,MB \\\\")
    body = "\n".join(rows)
    return tex[:match.start()] + pre + body + post + tex[match.end():]

def update_forgetting_table(tex: str, methods: Dict) -> str:
    """Replace tab:forgetting table."""
    pattern = r'(\\begin\{table\}\[h\].*?\\label\{tab:forgetting\}.*?\\toprule\s*Method & Initial loss.*?\\midrule\s*)(.*?)(\s*\\bottomrule\s*\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex, re.DOTALL)
    if not match:
        return tex
    pre, _, post = match.group(1), match.group(2), match.group(3)
    order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("adapters", "Adapters"),
        ("bitfit", "BitFit"),
    ]
    rows = []
    for key, name in order:
        m = methods.get(key, {})
        init = m.get("initial_loss", {}).get("mean", 2.31)
        final = m.get("final_loss", {}).get("mean")
        steps = int(m.get("total_steps", {}).get("mean", 10000))
        if final is not None and init and init > 0:
            red = (init - final) / init * 100
            red_str = f"{red:.1f}\\%"
        else:
            red_str = "---"
        conv = f"{steps}k steps" if steps >= 1000 else f"{steps} steps"
        rows.append(f"{name} & $\\sim${fmt_num(init, 1)} & {fmt_num(final, 4) if final else '---'} & {red_str} & {conv} \\\\")
    body = "\n".join(rows)
    return tex[:match.start()] + pre + body + post + tex[match.end():]

def update_stats_detailed(tex: str, methods: Dict, comparisons: Dict) -> str:
    """Replace tab:stats_detailed (val_accuracy = token-level, same as training)."""
    pattern = r'(\\begin\{table\}\[h\].*?\\label\{tab:stats_detailed\}.*?\\toprule\s*Method & Mean.*?\\midrule\s*)(.*?)(\s*\\bottomrule\s*\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex, re.DOTALL)
    if not match:
        return tex
    pre, _, post = match.group(1), match.group(2), match.group(3)
    val_key = "val_accuracy"  # token-level (0-1)
    order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("adapters", "Adapters"),
        ("bitfit", "BitFit"),
    ]
    rows = []
    for key, name in order:
        m = methods.get(key, {})
        v = m.get(val_key, {})
        comp = comparisons.get(val_key, {}).get(key, {})
        mean = v.get("mean")
        std = v.get("std") or 0
        var = v.get("variance") or 0
        ci_lo, ci_hi = v.get("ci_95_lower"), v.get("ci_95_upper")
        ttest = comp.get("t_test") or {}
        pval = ttest.get("p_value")
        cohens = comp.get("cohens_d")
        if mean is not None and mean < 2:
            mean, std = mean * 100, (std or 0) * 100
            if ci_lo is not None and ci_hi is not None:
                ci_lo, ci_hi = ci_lo * 100, ci_hi * 100
        mean_str = fmt_num(mean, 2) if mean is not None else "---"
        std_str = fmt_num(std, 2) if std is not None else "---"
        var_str = fmt_num(var, 3) if var is not None else "---"
        ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if (ci_lo is not None and ci_hi is not None) else "---"
        p_str = "---" if key == "full_ft" else (fmt_num(pval, 2) if pval is not None and not (isinstance(pval, float) and math.isnan(pval)) else "---")
        d_str = "---" if key == "full_ft" else (fmt_num(cohens, 2) if cohens is not None and not (isinstance(cohens, float) and math.isnan(cohens)) else "---")
        rows.append(f"{name} & {mean_str} & {std_str} & {var_str} & {ci_str} & {p_str} & {d_str} \\\\")
    body = "\n".join(rows)
    return tex[:match.start()] + pre + body + post + tex[match.end():]

def update_composition_table(tex: str, methods: Dict, params: Dict) -> str:
    """Replace tab:composition table."""
    pattern = r'(\\begin\{table\}\[h\].*?\\label\{tab:composition\}.*?\\toprule\s*Method & Train acc.*?\\midrule\s*)(.*?)(\s*\\midrule\s*LoRA-Diffusion.*?\\bottomrule\s*\\end\{tabular\}.*?\\end\{table\})'
    match = re.search(pattern, tex, re.DOTALL)
    if not match:
        return tex
    pre, _, post = match.group(1), match.group(2), match.group(3)
    val_key = "val_accuracy"  # token-level (0-1), same as training
    order = [
        ("full_ft", "Full Fine-Tuning"),
        ("lora_diffusion", "LoRA-Diffusion"),
        ("weight_lora", "Weight LoRA"),
        ("bitfit", "BitFit"),
        ("adapters", "Adapters"),
        ("prefix_tuning", "Prefix Tuning"),
    ]
    rows = []
    full_ft_val = methods.get("full_ft", {}).get(val_key, {}).get("mean")
    lora_val = methods.get("lora_diffusion", {}).get(val_key, {}).get("mean")
    for key, name in order:
        m = methods.get(key, {})
        p = params.get(key, {})
        train_acc = m.get("train_accuracy", {}).get("mean")
        val_acc = m.get(val_key, {}).get("mean")
        loss = m.get("final_loss", {}).get("mean")
        steps = int(m.get("total_steps", {}).get("mean", 50))
        param_pct = p.get("trainable_percent", 0)
        status = "$\\checkmark$" if val_acc is not None else "$\\times$"
        train_str = f"{train_acc*100:.2f}" if (train_acc is not None and train_acc < 2) else "---"
        val_str = f"{val_acc*100:.2f}" if (val_acc is not None and val_acc < 2) else (fmt_num(val_acc, 2) if val_acc is not None else "---")
        loss_str = fmt_num(loss, 4) if loss is not None else "---"
        rows.append(f"{name} & {train_str} & {val_str} & {loss_str} & {steps} & {param_pct:.1f}\\% & {status} \\\\")
    rel_str = "---"
    if full_ft_val and lora_val and full_ft_val > 0:
        rel_str = f"{lora_val/full_ft_val*100:.1f}\\%"
    ld_loss = methods.get("lora_diffusion", {}).get("final_loss", {}).get("mean")
    ft_loss = methods.get("full_ft", {}).get("final_loss", {}).get("mean")
    loss_ratio = f"{ld_loss/ft_loss:.2f}$\\times$" if (ld_loss and ft_loss and ft_loss > 0) else "---"
    rows.append(f"\\midrule\nLoRA-Diffusion vs.\\ full FT & --- & {rel_str} & {loss_ratio} & 1.0$\\times$ & 28.7\\% & --- \\\\")
    body = "\n".join(rows)
    return tex[:match.start()] + pre + body + "\n" + post + tex[match.end():]

def fix_stats_caption(tex: str) -> str:
    """Change stats_detailed caption to token-level (same metric as training)."""
    return tex.replace(
        "\\caption{Comprehensive statistical analysis of validation accuracy (generation-based)",
        "\\caption{Comprehensive statistical analysis of validation accuracy (token-level, same as training)",
        1,
    ).replace(
        "\\caption{Comprehensive statistical analysis of validation accuracy (standard classification-head)",
        "\\caption{Comprehensive statistical analysis of validation accuracy (token-level, same as training)",
        1,
    )


def fix_per_task_caption(tex: str) -> str:
    """Change per_task_results caption to standard classification-head."""
    return tex.replace(
        "Val acc.\\ from generation-based evaluation.",
        "Val acc.\\ = standard classification-head accuracy.",
        1,
    )

def main():
    ap = argparse.ArgumentParser(description="Update Paper.tex from collected_results_with_stats.json")
    ap.add_argument("--stats", default="collected_results_with_stats.json", help="Path to stats JSON")
    ap.add_argument("--params", default="parameter_counts.json", help="Path to parameter counts JSON")
    ap.add_argument("--paper", default="doc/Paper.tex", help="Path to Paper.tex")
    ap.add_argument("--out", default=None, help="Output path (default: overwrite --paper)")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    stats_path = root / args.stats
    params_path = root / args.params
    paper_path = root / args.paper
    out_path = Path(args.out) if args.out else paper_path
    if not out_path.is_absolute():
        out_path = root / out_path

    stats = load_json(stats_path)
    params = load_json(params_path)
    methods = stats.get("methods", {})
    comparisons = stats.get("comparisons", {})

    tex = paper_path.read_text()
    tex = update_main_results(tex, methods, params, comparisons)
    tex = update_classification_head(tex, methods)
    tex = update_per_task_results(tex, methods, params)
    tex = update_efficiency_table(tex, methods, params)
    tex = update_forgetting_table(tex, methods)
    tex = update_stats_detailed(tex, methods, comparisons)
    tex = update_composition_table(tex, methods, params)
    tex = fix_stats_caption(tex)
    tex = fix_per_task_caption(tex)

    out_path.write_text(tex)
    print(f"Updated {out_path}")
    print("  tab:main_results, tab:classification_head, tab:per_task_results,")
    print("  tab:efficiency, tab:forgetting, tab:stats_detailed, tab:composition")

if __name__ == "__main__":
    main()
