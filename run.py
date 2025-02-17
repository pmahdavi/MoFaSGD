#!/usr/bin/env python3
import os
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

def load_yaml_config(config_path: str) -> Dict[Any, Any]:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(base_config: Dict[Any, Any], override_config: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merge two configuration dictionaries."""
    merged = base_config.copy()
    merged.update(override_config)
    return merged

def get_final_config(optimizer: str, config_path: Optional[str] = None, config_override: Optional[str] = None) -> Dict[Any, Any]:
    """Get the final configuration after merging base config and overrides."""
    # Load base config for the optimizer
    base_config = {}
    default_config_path = Path(f'configs/optimizers/{optimizer}.yaml')
    if default_config_path.exists():
        base_config = load_yaml_config(str(default_config_path))
    
    # Load custom config if provided
    if config_path:
        custom_config = load_yaml_config(config_path)
        base_config = merge_configs(base_config, custom_config)
    
    # Apply JSON overrides if provided
    if config_override:
        override_config = json.loads(config_override)
        base_config = merge_configs(base_config, override_config)
    
    return base_config

def generate_run_name(optimizer: str, config: Dict[Any, Any]) -> str:
    """Generate a descriptive run name based on optimizer and config."""
    lr = config.get('lr', 'unknown_lr')
    
    # Add additional parameters based on optimizer type
    extra_params = []
    if optimizer in ['sgd', 'muon']:
        momentum = config.get('momentum', 'unknown_mom')
        extra_params.append(f"mom{momentum}")
    elif optimizer == 'lomuon':
        rank = config.get('rank', 'unknown_rank')
        beta = config.get('beta', 'unknown_beta')
        eta1 = config.get('eta1', 'unknown_eta1')
        eta2 = config.get('eta2', 'unknown_eta2')
        extra_params.extend([f"rank{rank}", f"beta{beta}", f"eta1_{eta1}", f"eta2_{eta2}"])
    elif optimizer in ['adam', 'adamw']:
        betas = config.get('betas', [0, 0])
        beta1, beta2 = betas if isinstance(betas, (list, tuple)) else (0, 0)
        if optimizer == 'adamw':
            wd = config.get('weight_decay', 0)
            extra_params.append(f"wd{wd}")
        extra_params.extend([f"b1_{beta1}", f"b2_{beta2}"])
    elif optimizer == 'galore':
        betas = config.get('betas', [0, 0])
        beta1, beta2 = betas if isinstance(betas, (list, tuple)) else (0, 0)
        wd = config.get('weight_decay', 0)
        group_params = config.get('group_params', {})
        rank = group_params.get('rank', 'unknown_rank')
        update_proj_gap = group_params.get('update_proj_gap', 'unknown_gap')
        scale = group_params.get('scale', 'unknown_scale')
        extra_params.extend([f"b1_{beta1}", f"b2_{beta2}", f"wd{wd}", f"rank_{rank}", f"update_proj_gap{update_proj_gap}", f"scale{scale}"])
    
    params_str = '_'.join(extra_params)
    return f"{optimizer}_lr{lr}_{params_str}"

def run_training(
    optimizer: str,
    config_path: Optional[str] = None,
    config_override: Optional[str] = None,
    num_gpus: int = 1
) -> None:
    """Run the training script with the specified configuration."""
    
    # Validate optimizer choice
    valid_optimizers = ['muon', 'adam', 'adamw', 'sgd', 'galore', 'lomuon']
    if optimizer.lower() not in valid_optimizers:
        raise ValueError(f"Invalid optimizer. Must be one of: {', '.join(valid_optimizers)}")
    
    # Get final config for run name
    final_config = get_final_config(optimizer, config_path, config_override)
    run_name = generate_run_name(optimizer, final_config)
    
    # Build command
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        "train_gpt.py",
        "--optimizer", optimizer,
        "--run-name", run_name,  # Pass run name to train_gpt.py
    ]
    
    # Add config path if specified
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cmd.extend(["--config-path", config_path])
    
    # Add config override if specified
    if config_override:
        try:
            # Validate JSON
            json.loads(config_override)
            cmd.extend(["--config", config_override])
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in config override")
    
    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:."
    
    # Run the command
    print(f"Starting run: {run_name}")
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run GPT training with different optimizers and configurations')
    parser.add_argument('--optimizer', type=str, default='muon',
                      choices=['muon', 'adam', 'adamw', 'sgd', 'galore', 'lomuon'],
                      help='Optimizer to use')
    parser.add_argument('--config-path', type=str,
                      help='Path to YAML config file')
    parser.add_argument('--config', type=str,
                      help='JSON string with config overrides')
    parser.add_argument('--num-gpus', type=int, default=1,
                      help='Number of GPUs to use')
    parser.add_argument('--list-configs', action='store_true',
                      help='List available optimizer configurations')
    
    args = parser.parse_args()
    
    # Handle listing configs
    if args.list_configs:
        config_dir = Path('configs/optimizers')
        if config_dir.exists():
            print("Available optimizer configurations:")
            for config_file in config_dir.glob('*.yaml'):
                print(f"\n{config_file.stem}:")
                config = load_yaml_config(str(config_file))
                print(yaml.dump(config, indent=2))
        return
    
    try:
        run_training(
            optimizer=args.optimizer,
            config_path=args.config_path,
            config_override=args.config,
            num_gpus=args.num_gpus
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main() 