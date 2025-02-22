import math
import torch
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    ExponentialLR
)

def get_original_scheduler(optimizer, num_iterations, optimizer_config):
    """Original scheduler from train_gpt.py"""
    def get_lr(it):
        t = 1 - it / num_iterations
        cooldown_frac = optimizer_config.get('cooldown_frac', 0.4)
        if t >= cooldown_frac:
            return 1.0
        else:
            return t / cooldown_frac
    return LambdaLR(optimizer, get_lr)

def get_original_warmup_scheduler(optimizer, num_iterations, optimizer_config):
    """Original scheduler with warmup"""
    lr_warmup_steps = optimizer_config.get('lr_warmup_steps', 100)
    cooldown_frac = optimizer_config.get('cooldown_frac', 0.4)
    
    def get_lr(it):
        if it < lr_warmup_steps:
            return it / lr_warmup_steps
        
        t = 1 - it / num_iterations
        if t >= cooldown_frac:
            return 1.0
        else:
            return t / cooldown_frac
    return LambdaLR(optimizer, get_lr)

def get_cosine_warmup_scheduler(optimizer, num_iterations, optimizer_config):
    """Cosine decay with warmup"""
    lr_warmup_steps = optimizer_config.get('lr_warmup_steps', 100)
    def get_lr(it):
        if it < lr_warmup_steps:
            return it / lr_warmup_steps
        else:
            progress = (it - lr_warmup_steps) / (num_iterations - lr_warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, get_lr)

def get_cosine_scheduler(optimizer, num_iterations, optimizer_config):
    """Cosine annealing"""
    return CosineAnnealingLR(
        optimizer,
        T_max=num_iterations,
        eta_min=optimizer_config.get('min_lr', 0)
    )

def get_exponential_scheduler(optimizer, num_iterations, optimizer_config):
    """Exponential decay"""
    gamma = optimizer_config.get('decay_rate', 0.95)
    return ExponentialLR(optimizer, gamma=gamma)

SCHEDULER_REGISTRY = {
    'original': get_original_scheduler,
    'original_warmup': get_original_warmup_scheduler,
    'cosine_warmup': get_cosine_warmup_scheduler,
    'cosine': get_cosine_scheduler,
    'exponential': get_exponential_scheduler,
}

def create_scheduler(name, optimizer, num_iterations, optimizer_config):
    """Factory function to create scheduler based on name"""
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler: {name}. Available schedulers: {list(SCHEDULER_REGISTRY.keys())}")
    return SCHEDULER_REGISTRY[name](optimizer, num_iterations, optimizer_config) 