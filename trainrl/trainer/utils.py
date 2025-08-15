import os
import torch
import torch.nn.functional as F
import torch.distributed as dist

import signal
import sys

def setup_signal_handlers(rank):
    """Setup signal handlers for clean shutdown."""
    def shutdown_handler(signum, frame):
        print(f"[Rank {rank}] Received signal {signum}. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

def get_log_probs(logits, actions):
    """
    Compute log probabilities for given actions from logits.
    Ensures device consistency between logits and actions.
    
    Args:
        logits: [B, T, V] - model output logits
        actions: [B, T] - action tokens
    
    Returns:
        action_log_probs: [B, T] - log probabilities for the actions
    """
    # CRITICAL FIX: Ensure both tensors are on the same device
    if logits.device != actions.device:
        actions = actions.to(logits.device)
    
    # Ensure actions has the right dtype
    if actions.dtype != torch.long:
        actions = actions.long()
    
    # Add numerical stability
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    return action_log_probs

def get_transformer_layer_classes(model):
    """
    Dynamically find all transformer layer classes in the core model.
    Assumes model is wrapped in ModelShell and has `core_model`.
    """
    layer_classes = set()

    core_model = model.core_model if hasattr(model, "core_model") else model

    for module in core_model.modules():
        cls = type(module)
        if cls.__name__.lower().endswith("decoderlayer") or cls.__name__.lower().endswith("layer"):
            layer_classes.add(cls)

    return layer_classes

def init_distributed_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Distributed setup complete on rank {rank}/{world_size}")

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def ensure_device_consistency(*tensors, target_device=None):
    """
    Ensure all tensors are on the same device.
    
    Args:
        *tensors: Variable number of tensors
        target_device: Device to move tensors to. If None, uses device of first tensor.
    
    Returns:
        List of tensors all on the same device
    """
    if not tensors:
        return []
    
    # Determine target device
    if target_device is None:
        for tensor in tensors:
            if hasattr(tensor, 'device'):
                target_device = tensor.device
                break
    
    # Move all tensors to target device
    result = []
    for tensor in tensors:
        if hasattr(tensor, 'to'):
            result.append(tensor.to(target_device))
        else:
            result.append(tensor)
    
    return result