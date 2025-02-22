"""
Utility functions for model analysis and logging.
"""

def print_model_parameters(model, print0_fn):
    """
    Print detailed information about model parameters.
    
    Args:
        model: The PyTorch model to analyze
        print0_fn: Function to use for printing (should handle both console and file logging)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print0_fn("\nModel Parameter Details:", console=True)
    print0_fn("=" * 50, console=True)
    print0_fn(f"Total Parameters: {total_params:,}", console=True)
    print0_fn(f"Trainable Parameters: {trainable_params:,}", console=True)
    
    # Memory usage estimation
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print0_fn(f"Estimated Model Size: {size_mb:.2f} MB", console=True)
    
    print0_fn("\nParameters by Component:", console=True)
    print0_fn("-" * 50, console=True)
    
    # Detailed breakdown by component
    component_params = {
        'Embedding Layer': sum(p.numel() for p in model.embed.parameters()),
        'Value Embeddings': sum(p.numel() for p in model.value_embeds.parameters()),
        'Transformer Blocks': sum(p.numel() for p in model.blocks.parameters()),
        'Language Model Head': sum(p.numel() for p in model.lm_head.parameters())
    }
    
    for component, params in component_params.items():
        print0_fn(f"{component}: {params:,} parameters ({params/total_params*100:.2f}%)", console=True)
    
    print0_fn("\nParameters by Layer Type:", console=True)
    print0_fn("-" * 50, console=True)
    layer_types = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0] if '.' in name else 'other'
        if layer_type not in layer_types:
            layer_types[layer_type] = {'params': 0, 'count': 0}
        layer_types[layer_type]['params'] += param.numel()
        layer_types[layer_type]['count'] += 1
    
    for layer_type, info in layer_types.items():
        print0_fn(f"{layer_type}: {info['params']:,} parameters in {info['count']} tensors", console=True)
    
    print0_fn("\nParameters by Shape:", console=True)
    print0_fn("-" * 50, console=True)
    for name, param in model.named_parameters():
        print0_fn(f"{name}: {tuple(param.shape)} = {param.numel():,} parameters", console=True)
    
    print0_fn("\nGradient Status:", console=True)
    print0_fn("-" * 50, console=True)
    for name, param in model.named_parameters():
        print0_fn(f"{name}: requires_grad = {param.requires_grad}", console=True)
    
    print0_fn("=" * 50 + "\n", console=True)

def get_model_size_stats(model):
    """
    Get model size statistics in a dictionary format.
    
    Args:
        model: The PyTorch model to analyze
    
    Returns:
        dict: Dictionary containing model size statistics
    """
    stats = {}
    
    # Basic parameter counts
    stats['total_params'] = sum(p.numel() for p in model.parameters())
    stats['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory usage
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    stats['model_size_mb'] = (param_size + buffer_size) / 1024 / 1024
    
    # Component-wise breakdown
    stats['component_params'] = {
        'embedding_layer': sum(p.numel() for p in model.embed.parameters()),
        'value_embeddings': sum(p.numel() for p in model.value_embeds.parameters()),
        'transformer_blocks': sum(p.numel() for p in model.blocks.parameters()),
        'lm_head': sum(p.numel() for p in model.lm_head.parameters())
    }
    
    return stats 