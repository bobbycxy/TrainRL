import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
    StateDictType,
    FullStateDictConfig,
    ShardingStrategy,
)
import torch.distributed as dist
from copy import deepcopy
import numpy as np
import os
from functools import partial

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)
        # Proper initialization for value head
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)
    
    def forward(self, hidden_state):
        return self.value(hidden_state).squeeze(-1) # [B, T]

def get_transformer_layer_classes(model):
    """Extract transformer layer classes for FSDP wrapping - adapted from your utils"""
    transformer_layer_classes = set()
    
    def find_transformer_layers(module):
        # Common transformer layer class name patterns
        layer_patterns = [
            'DecoderLayer', 'EncoderLayer', 'TransformerLayer', 'Block', 'Layer'
        ]
        
        for name, child in module.named_children():
            class_name = child.__class__.__name__
            if any(pattern in class_name for pattern in layer_patterns):
                transformer_layer_classes.add(child.__class__)
            find_transformer_layers(child)
    
    find_transformer_layers(model)
    return transformer_layer_classes

def get_fsdp_wrap_policy(model):
    """Get FSDP wrap policy using transformer layer detection"""
    transformer_layer_classes = get_transformer_layer_classes(model)
    if transformer_layer_classes:
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_classes
        )
    else:
        # Fallback to size-based wrapping
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        return partial(size_based_auto_wrap_policy, min_num_params=1e6)

class FSDPWrapper:
    """FSDP wrapper adapted from your distwrapper.py"""
    def __init__(self, model, rank, mixed_precision_config=None):
        self.rank = rank
        self.original_model = model
        self.wrapped_model = self._wrap_model(mixed_precision_config)
    
    def _wrap_model(self, mixed_precision_config=None):
        print(f"[Rank {self.rank}] Using FSDP for model wrapping...")
        
        auto_wrap_policy = get_fsdp_wrap_policy(self.original_model)
        
        fsdp_kwargs = {
            'auto_wrap_policy': auto_wrap_policy,
            'device_id': self.rank,
            'sync_module_states': True,
            'use_orig_params': True,
            'sharding_strategy': ShardingStrategy.FULL_SHARD,
            'forward_prefetch': True,
        }
        
        if mixed_precision_config:
            fsdp_kwargs['mixed_precision'] = mixed_precision_config
        
        return FSDP(self.original_model, **fsdp_kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.wrapped_model, name)

class PPOTrainer:
    def __init__(
        self, model, tokenizer, value_head, optimizer, reward_model,
        gamma=0.99, lam=0.95, clip_eps=0.2, c1=0.5, c2=0.01,
        ppo_epochs=4, batch_size=4, max_grad_norm=1.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.value_head = value_head
        self.optimizer = optimizer
        self.reward_model = reward_model
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        
        # Initialize mixed precision config
        self.mp_config = self._init_mixed_precision()
        
        # Create separate reference model for production PPO
        self.ref_model = self._create_reference_model()
        
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] PPO Trainer initialized with separate reference model")

    def _create_reference_model(self):
        """Create a separate reference model using state dict approach - FSDP compatible"""
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Creating reference model...")
        
        # We need to create a reference model without FSDP wrapping first
        # Then copy the state dict and wrap it with FSDP
        
        # Method 1: Try to create a copy via state dict
        try:
            # Get the original unwrapped model architecture
            if hasattr(self.model, 'wrapped_model'):
                # If it's wrapped, we need to get the original architecture
                base_model_name = "Qwen/Qwen3-1.7B"  # We know this from setup
                
                from transformers import AutoModelForCausalLM
                ref_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, 
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map=None,
                    attn_implementation="eager"
                )
                
                # Move to same device
                device = next(self.model.parameters()).device
                ref_model = ref_model.to(device)
                
                # Enable gradient checkpointing to match main model
                if hasattr(ref_model, "gradient_checkpointing_enable"):
                    ref_model.gradient_checkpointing_enable()
                
                # Now wrap with FSDP using same configuration
                mixed_precision_config = torch.distributed.fsdp.MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                )
                
                rank = dist.get_rank() if dist.is_initialized() else 0
                ref_wrapper = FSDPWrapper(ref_model, rank, mixed_precision_config)
                ref_model = ref_wrapper.wrapped_model
                
                # Copy state from main model to reference model
                self._copy_model_state(self.model, ref_model)
                
            else:
                # Non-FSDP case - should not happen in our setup, but good fallback
                ref_model = deepcopy(self.model)
                
            # Set to eval mode and freeze parameters  
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            
            print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Reference model created successfully")
            return ref_model
            
        except Exception as e:
            print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Error creating reference model: {e}")
            # Fallback: return None and handle in _get_old_policy_logits
            return None

    def _copy_model_state(self, source_model, target_model):
        """Copy state from source to target model - FSDP compatible"""
        try:
            is_fsdp = isinstance(getattr(source_model, "wrapped_model", source_model), FSDP)
            
            if is_fsdp:
                # Use FSDP state dict copying
                with FSDP.state_dict_type(
                    source_model,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
                ):
                    source_state = source_model.state_dict()
                
                with FSDP.state_dict_type(
                    target_model,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
                ):
                    target_model.load_state_dict(source_state)
            else:
                # Non-FSDP case
                target_model.load_state_dict(source_model.state_dict())
                
            print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model state copied successfully")
        except Exception as e:
            print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Warning: Could not copy model state: {e}")

    def _init_mixed_precision(self, use_tf32=True):
        """Mixed precision setup adapted from your base_trainer.py"""
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Initializing mixed precision...")
        
        device = next(self.model.parameters()).device
        
        # Select precision based on hardware capabilities
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )

        # Enable TensorFloat-32 for faster matrix multiplications on CUDA
        if device.type == "cuda" and use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Initialize gradient scaler (only for float16)
        scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

        # Create autocast context manager
        ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print("Mixed Precision Setup:")
            print(f"- Device: {device}")
            print(f"- Precision: {dtype}")
            print(f"- TF32 Enabled: {use_tf32}")
            print(f"- Gradient Scaler Enabled: {scaler.is_enabled()}")

        return {"ctx": ctx, "scaler": scaler, "dtype": dtype}

    def _save_old_model_state(self):
        """Save model state for reference policy - production ready with proper FSDP handling"""
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Saving old model state...")
        
        is_fsdp = isinstance(getattr(self.model, "wrapped_model", self.model), FSDP)
        
        if is_fsdp:
            # Use the exact same pattern as your base_trainer._save_model()
            with FSDP.state_dict_type(
                self.model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                if dist.get_rank() == 0:
                    self.old_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    print(f"[Rank 0] Saved {len(self.old_model_state)} parameters for reference policy")
                else:
                    self.old_model_state = None
        else:
            # Non-FSDP case
            if dist.get_rank() == 0:
                raw_state_dict = self.model.state_dict()
                self.old_model_state = {
                    k.replace("module.", ""): v.cpu().clone() for k, v in raw_state_dict.items()
                }
            else:
                self.old_model_state = None
        
        # Broadcast a flag to all ranks that state is saved
        if dist.is_initialized():
            state_saved = torch.tensor(1 if self.old_model_state is not None else 0, device=f'cuda:{dist.get_rank()}')
            dist.broadcast(state_saved, src=0)
            if dist.get_rank() != 0:
                self.old_model_state = state_saved.item() == 1  # Just a flag for non-rank0
        
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Old model state saved successfully")

    def _get_old_policy_logits(self, input_ids, attention_mask):
        """Get reference policy logits using separate reference model - production ready"""
        if self.ref_model is None:
            # Fallback: use current model but detach (not ideal but safe)
            with torch.no_grad():
                with self.mp_config["ctx"]:
                    outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    old_logits = outputs.logits.detach().clone()
                    old_hidden = outputs.hidden_states[-1].detach().clone()
            return old_logits, old_hidden
        
        # Use separate reference model
        with torch.no_grad():
            with self.mp_config["ctx"]:
                ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                old_logits = ref_outputs.logits
                old_hidden = ref_outputs.hidden_states[-1]
        
        return old_logits, old_hidden

    def compute_gae(self, rewards, values, mask):
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Pad values to handle the last timestep properly
        values_padded = F.pad(values, (0, 1), mode='constant', value=0)
        
        for t in reversed(range(T)):
            mask_t = mask[:, t]
            # Use next value for bootstrap (or 0 for terminal states)
            next_value = values_padded[:, t + 1] * mask_t  # Zero out if terminal
            delta = rewards[:, t] + self.gamma * next_value - values_padded[:, t]
            last_gae = delta + self.gamma * self.lam * last_gae * mask_t
            advantages[:, t] = last_gae
            
        returns = advantages + values_padded[:, :-1]
        return advantages.detach(), returns.detach()
    
    def _get_log_probs(self, logits, actions):
        # Add numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        return action_log_probs
    
    def _compute_rewards(self, input_ids, prompt_mask):
        with torch.no_grad():
            scores = self.reward_model(input_ids)
            # Apply reward only to generated tokens
            rewards = scores * prompt_mask
            # Normalize rewards per batch to reduce variance
            if rewards.numel() > 0:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards
    
    def train(self, input_ids, attention_mask, prompt_mask, actions):
        """Training loop with FSDP support"""
        device = input_ids.device
        
        with torch.no_grad():
            # Compute rewards
            rewards = self._compute_rewards(input_ids, prompt_mask)
            
            # Get old policy log probs and values
            old_logits, old_hidden = self._get_old_policy_logits(input_ids, attention_mask)
            old_log_probs = self._get_log_probs(old_logits, actions)
            old_values = self.value_head(old_hidden)

        for epoch in range(self.ppo_epochs):
            dataset = TensorDataset(input_ids, attention_mask, prompt_mask, actions, rewards, old_log_probs, old_values)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for batch in dataloader:
                input_ids_b, attn_mask_b, prompt_mask_b, actions_b, rewards_b, old_log_probs_b, old_values_b = batch

                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass with mixed precision
                with self.mp_config["ctx"]:
                    outputs = self.model(input_ids_b, attention_mask=attn_mask_b, output_hidden_states=True)
                    logits = outputs.logits
                    log_probs = self._get_log_probs(logits, actions_b)

                    # Value predictions
                    hidden_states = outputs.hidden_states[-1]
                    values = self.value_head(hidden_states)

                    # GAE computation
                    advantages, returns = self.compute_gae(rewards=rewards_b, values=values, mask=prompt_mask_b)
                    
                    # Normalize advantages
                    if advantages.numel() > 0:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # PPO policy loss
                    ratios = torch.exp(log_probs - old_log_probs_b)
                    ratios = torch.clamp(ratios, 0.1, 10.0)
                    
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2)
                    policy_loss = (policy_loss * prompt_mask_b).sum() / (prompt_mask_b.sum() + 1e-8)

                    # Value loss with clipping
                    value_pred_clipped = old_values_b + torch.clamp(
                        values - old_values_b, -self.clip_eps, self.clip_eps
                    )
                    value_loss1 = (values - returns) ** 2
                    value_loss2 = (value_pred_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                    value_loss = (value_loss * prompt_mask_b).sum() / (prompt_mask_b.sum() + 1e-8)

                    # Entropy loss
                    entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1)
                    entropy = (entropy * prompt_mask_b).sum() / (prompt_mask_b.sum() + 1e-8)

                    # Total loss
                    total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

                # Backward pass
                if torch.isfinite(total_loss):
                    self.mp_config["scaler"].scale(total_loss).backward()
                    
                    # Gradient clipping
                    self.mp_config["scaler"].unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.value_head.parameters()), 
                        self.max_grad_norm
                    )
                    
                    self.mp_config["scaler"].step(self.optimizer)
                    self.mp_config["scaler"].update()

        # Aggregate metrics across ranks like in your base_trainer
        metrics = {
            "loss": total_loss.item() if torch.isfinite(total_loss) else float('inf'),
            "policy_loss": policy_loss.item() if torch.isfinite(policy_loss) else float('inf'),
            "value_loss": value_loss.item() if torch.isfinite(value_loss) else float('inf'),
            "entropy": entropy.item() if torch.isfinite(entropy) else 0.0,
            "mean_return": returns.mean().item() if torch.isfinite(returns).all() else 0.0
        }
        
        # Average metrics across ranks
        if dist.is_initialized():
            for key in metrics:
                if metrics[key] != float('inf'):
                    tensor = torch.tensor(metrics[key], device=device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                    metrics[key] = tensor.item()
        
        return metrics

    def update_old_policy(self):
        """Update the reference policy by copying current model weights - production ready"""
        if self.ref_model is None:
            print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] No reference model available, skipping update")
            return
            
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Updating reference policy...")
        self._copy_model_state(self.model, self.ref_model)

def aggregate_value(value):
    """Aggregate values across ranks - from your utils"""
    if not dist.is_initialized():
        return value
    
    tensor = torch.tensor(value).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()

from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HFRewardModel(nn.Module):
    def __init__(self, model_name="OpenAssistant/reward-model-deberta-v3-large"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def forward(self, input_ids):
        with torch.no_grad():
            # Decode from token ids to text
            decoded = [self.tokenizer.decode(x, skip_special_tokens=True) for x in input_ids]
            enc = self.tokenizer(decoded, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(input_ids.device) for k, v in enc.items()}
            
            outputs = self.model(**enc)
            scores = outputs.logits.squeeze(-1)  # [B]
            # Expand to sequence length and normalize
            scores = torch.tanh(scores) * 5.0  # Bound rewards to reasonable range
            return scores.unsqueeze(1).expand(-1, input_ids.shape[1])  # [B, T]

def setup_distributed():
    """Initialize distributed training"""
    print("Setting up distributed training...")
    
    if "RANK" in os.environ:
        # SLURM or torchrun environment
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Found distributed environment: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        # Single node training
        rank = 0
        world_size = 1
        local_rank = 0
        print("Single node training detected")
    
    if world_size > 1:
        print(f"Initializing process group for rank {rank}...")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        print(f"Process group initialized for rank {rank}")
        
        # Add a barrier to ensure all processes are ready
        print(f"Rank {rank} waiting at barrier...")
        dist.barrier()
        print(f"Rank {rank} passed barrier")
    
    return rank, world_size, local_rank

def setup_training():
    rank, world_size, local_rank = setup_distributed()
    
    print(f"[Rank {rank}] Starting setup...")
    
    base = "Qwen/Qwen3-1.7B"
    print(f"[Rank {rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    
    # Fix padding issue
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print(f"[Rank {rank}] Loading model...")
    # Load model without device_map for FSDP
    model = AutoModelForCausalLM.from_pretrained(
        base, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,  # Important: don't use device_map with FSDP
        attn_implementation="eager"
    )
    
    print(f"[Rank {rank}] Model loaded, moving to device...")
    # Move model to correct device before FSDP wrapping
    model = model.to(f'cuda:{local_rank}')
    
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print(f"[Rank {rank}] Gradient checkpointing enabled")
    
    print(f"[Rank {rank}] Creating mixed precision config...")
    # Create mixed precision config
    mixed_precision_config = torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,  # Use float32 for reductions
        buffer_dtype=torch.bfloat16,
    )
    
    print(f"[Rank {rank}] Wrapping model with FSDP...")
    # Wrap model with FSDP using your wrapper
    model_wrapper = FSDPWrapper(model, rank, mixed_precision_config)
    model = model_wrapper.wrapped_model
    
    print(f"[Rank {rank}] Creating value head...")
    # Create and wrap value head
    value_head = ValueHead(model.config.hidden_size).to(f'cuda:{local_rank}')
    print(f"[Rank {rank}] Wrapping value head with FSDP...")
    value_head_wrapper = FSDPWrapper(value_head, rank, mixed_precision_config)
    value_head = value_head_wrapper.wrapped_model

    print(f"[Rank {rank}] Setting up optimizer...")
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(value_head.parameters()), 
        lr=1e-5,
        weight_decay=0.01
    )
    
    print(f"[Rank {rank}] Loading reward model...")
    # Load reward model
    reward_model = HFRewardModel()
    if torch.cuda.is_available():
        reward_model = reward_model.cuda(local_rank)

    print(f"[Rank {rank}] Creating PPO trainer...")
    trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        value_head=value_head,
        optimizer=optimizer,
        reward_model=reward_model,
        clip_eps=0.2,
        c1=0.5,
        c2=0.01,
        ppo_epochs=2,
        batch_size=2,
        max_grad_norm=0.5
    )
    
    print(f"[Rank {rank}] Setup complete!")
    return trainer, tokenizer, rank, world_size, local_rank

from datasets import load_dataset
from random import choices
import torch.distributed as dist

def generate_and_train(prompts, trainer, tokenizer, max_new_tokens=32, rank=0):
    """Generate responses and train with PPO - with FSDP generation support"""
    
    # Tokenize input prompts
    encodings = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=256
    )
    
    device = next(trainer.model.parameters()).device
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    prompt_lens = attention_mask.sum(dim=1).tolist()

    # Generate responses with proper FSDP context
    with torch.no_grad():
        trainer.model.eval()
        
        # Use FSDP summon_full_params for generation
        is_fsdp = isinstance(getattr(trainer.model, "wrapped_model", trainer.model), FSDP)
        
        if is_fsdp:
            with trainer.model.summon_full_params(trainer.model):
                generation = trainer.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
        else:
            generation = trainer.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )
        
        generated_ids = generation.sequences[:, input_ids.shape[1]:]
        trainer.model.train()

    # Combine prompt + generation
    full_input_ids = torch.cat([input_ids, generated_ids], dim=1)
    full_attention_mask = torch.cat([
        attention_mask,
        torch.ones_like(generated_ids)
    ], dim=1)

    # Create masks
    actions = full_input_ids
    B, T = full_input_ids.shape
    prompt_mask = torch.zeros((B, T), dtype=torch.float32, device=device)
    for i, prompt_len in enumerate(prompt_lens):
        prompt_mask[i, prompt_len:] = 1.0

    # Train PPO
    try:
        metrics = trainer.train(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            prompt_mask=prompt_mask,
            actions=actions,
        )
        return metrics
    except Exception as e:
        if rank == 0:
            print(f"Training error: {e}")
        return {"loss": float('inf'), "policy_loss": float('inf'), "value_loss": float('inf'), "entropy": 0.0, "mean_return": 0.0}

# Main training loop
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    
    trainer, tokenizer, rank, world_size, local_rank = setup_training()
    
    if rank == 0:
        print(f"Starting distributed training on {world_size} GPUs")
    
    # Load dataset (only on rank 0)
    if rank == 0:
        try:
            dataset = load_dataset("OpenAssistant/oasst1", split="train")
            prompt_texts = [
                example["text"] for example in dataset
                if example.get("text") and len(example["text"]) < 200
            ][:1000]
        except:
            print("LOADING FALLBACK")
            prompt_texts = [
                "What is the capital of France?",
                "Explain photosynthesis in simple terms.",
                "Write a short story about a robot.",
                "How do you make chocolate chip cookies?",
                "What causes the seasons to change?"
            ] * 200
        
        print(f"Loaded {len(prompt_texts)} prompts")
    else:
        prompt_texts = None
    
    # Broadcast prompts to all ranks
    if world_size > 1:
        # Use torch.distributed to properly broadcast the data
        if rank == 0:
            num_prompts = len(prompt_texts)
        else:
            num_prompts = 0
        
        # Broadcast number of prompts first
        num_prompts_tensor = torch.tensor(num_prompts, device=f'cuda:{local_rank}')
        dist.broadcast(num_prompts_tensor, src=0)
        num_prompts = num_prompts_tensor.item()
        
        # For simplicity, use the same prompts on all ranks
        if rank != 0:
            prompt_texts = [
                "What is the capital of France?",
                "Explain photosynthesis in simple terms.", 
                "Write a short story about a robot.",
                "How do you make chocolate chip cookies?",
                "What causes the seasons to change?"
            ] * 200
    
    num_steps = 100
    batch_size = 2
    max_new_tokens = 32

    for step in range(num_steps):
        # Sample prompts
        prompts_batch = choices(prompt_texts, k=batch_size)

        # Run PPO update
        metrics = generate_and_train(
            prompts=prompts_batch,
            trainer=trainer,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            rank=rank
        )

        if rank == 0:
            print(f"[Step {step}] Loss: {metrics['loss']:.2f}, Policy: {metrics['policy_loss']:.2f}, "
                  f"Value: {metrics['value_loss']:.2f}, Entropy: {metrics['entropy']:.4f}, Return: {metrics['mean_return']:.2f}")

        # Update old policy less frequently
        if step % 8 == 0 and step > 0:
            trainer.update_old_policy()
            if rank == 0:
                print("Updated old policy")
                
        # Early stopping if loss explodes
        if metrics['loss'] > 1000:
            if rank == 0:
                print("Loss too high, stopping training")
            break
    
    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()