"""Configuration management utilities."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dictionaries
        
    Returns:
        Merged configuration
    """
    def merge_dicts(base: Dict, override: Dict) -> Dict:
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result
    
    if len(configs) == 0:
        return {}
    
    merged = deepcopy(configs[0])
    for config in configs[1:]:
        merged = merge_dicts(merged, config)
    
    return merged


def load_config(
    base_config_path: Path,
    task_name: Optional[str] = None,
    method_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load configuration with task and method overrides.
    
    Args:
        base_config_path: Path to base config file
        task_name: Optional task name to load task-specific config
        method_name: Optional method name to load method-specific config
        overrides: Optional dictionary of override values
        
    Returns:
        Complete merged configuration
    """
    # Load base config
    base_config = load_yaml(base_config_path)
    logger.info(f"Loaded base config from {base_config_path}")
    
    configs_to_merge = [base_config]
    
    # Load task config if specified
    if task_name:
        task_config_path = base_config_path.parent / "tasks" / f"{task_name}.yaml"
        if task_config_path.exists():
            task_config = load_yaml(task_config_path)
            configs_to_merge.append(task_config)
            logger.info(f"Loaded task config from {task_config_path}")
        else:
            logger.warning(f"Task config not found: {task_config_path}")
    
    # Load method config if specified
    if method_name:
        method_config_path = base_config_path.parent / "methods" / f"{method_name}.yaml"
        if method_config_path.exists():
            method_config = load_yaml(method_config_path)
            configs_to_merge.append(method_config)
            logger.info(f"Loaded method config from {method_config_path}")
        else:
            logger.warning(f"Method config not found: {method_config_path}")
    
    # Add overrides
    if overrides:
        configs_to_merge.append(overrides)
        logger.info("Applied command-line overrides")
    
    # Merge all configs
    final_config = merge_configs(*configs_to_merge)
    
    return final_config


def dict_to_namespace(config: Dict[str, Any]) -> Any:
    """Convert nested dictionary to nested namespace object."""
    from argparse import Namespace
    
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = dict_to_namespace(value)
        return Namespace(**config)
    elif isinstance(config, list):
        return [dict_to_namespace(item) for item in config]
    else:
        return config


def parse_override_string(override_str: str) -> tuple:
    """
    Parse override string like 'training.learning_rate=1e-4' into key path and value.
    
    Args:
        override_str: String in format 'key.subkey=value'
        
    Returns:
        (key_path, value) tuple where key_path is list of keys
    """
    if "=" not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected key=value")
    
    key_path_str, value_str = override_str.split("=", 1)
    key_path = key_path_str.split(".")
    
    # Try to infer type
    value: Any
    if value_str.lower() == "true":
        value = True
    elif value_str.lower() == "false":
        value = False
    elif value_str.lower() == "none" or value_str.lower() == "null":
        value = None
    else:
        # Try int, then float, then keep as string
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
    
    return key_path, value


def apply_override(config: Dict[str, Any], key_path: list, value: Any):
    """Apply a single override to config dictionary."""
    current = config
    for key in key_path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[key_path[-1]] = value


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
