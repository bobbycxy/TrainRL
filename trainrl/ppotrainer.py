import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
from copy import deepcopy
import numpy as np
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)
        # Proper initialization for value head
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)
    
    def forward(self, hidden_state):
        return self.value(hidden_state).squeeze(-1) # [B, T]

class PPOTrainerFSDP:
    def __init__(
        self, model, tokenizer, value_head, optimizer, reward_model,
        gamma=0.99, lam=0.95, clip_eps=0.2, c1=0.5, c2=0.01,
        ppo_epochs=4, batch_size=4, max_grad_norm=1.0,
        rank=0, world_size=1, use_mixed_precision=True
    ):
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.rank = rank
        self.world_size = world_size
        
        # Setup mixed precision if requested
        self.mixed_precision = None
        if use_mixed_precision:
            self.mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        
        # Wrap models with FSDP
        self.model = model
        self.value_head = value_head
        self.optimizer = optimizer
        self.reward_model = reward_model
        
        # SOLUTION 1: Don't create old_model in __init__
        # We'll create it later in setup_training_fsdp and assign it
        self.old_model = None

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
        """
        input_ids: [B, T] (prompt + completion)
        attention_mask: [B, T] 
        prompt_mask: [B, T] where 1.0 = generated token, 0.0 = prompt token
        actions: same as input_ids (for autoregressive LM)
        """
        device = input_ids.device
        
        with torch.no_grad():
            # Compute rewards
            rewards = self._compute_rewards(input_ids, prompt_mask)
            
            # Get old policy log probs
            old_outputs = self.old_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            old_logits = old_outputs.logits
            old_log_probs = self._get_log_probs(old_logits, actions)
            
            # Get old values for the baseline
            old_values = self.value_head(old_outputs.hidden_states[-1])

        for epoch in range(self.ppo_epochs):
            dataset = TensorDataset(input_ids, attention_mask, prompt_mask, actions, rewards, old_log_probs, old_values)
            
            # Use DistributedSampler for multi-GPU training
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank) if self.world_size > 1 else None
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=(sampler is None),
                sampler=sampler
            )
            
            if sampler:
                sampler.set_epoch(epoch)

            for batch in dataloader:
                input_ids_b, attn_mask_b, prompt_mask_b, actions_b, rewards_b, old_log_probs_b, old_values_b = batch

                # Forward pass
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

                # PPO policy loss - FIXED CLIPPING BOUNDS
                ratios = torch.exp(log_probs - old_log_probs_b)
                # Clamp ratios to prevent extreme values
                ratios = torch.clamp(ratios, 0.1, 10.0)
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2)
                policy_loss = (policy_loss * prompt_mask_b).sum() / (prompt_mask_b.sum() + 1e-8)

                # Value loss with clipping for stability
                value_pred_clipped = old_values_b + torch.clamp(
                    values - old_values_b, -self.clip_eps, self.clip_eps
                )
                value_loss1 = (values - returns) ** 2
                value_loss2 = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                value_loss = (value_loss * prompt_mask_b).sum() / (prompt_mask_b.sum() + 1e-8)

                # Entropy loss for exploration
                entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1)
                entropy = (entropy * prompt_mask_b).sum() / (prompt_mask_b.sum() + 1e-8)

                # Total loss
                total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

                # Check for NaN/inf before backward pass
                if not torch.isfinite(total_loss):
                    if self.rank == 0:
                        print(f"Warning: Non-finite loss detected: {total_loss.item()}")
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping - FSDP compatible
                if isinstance(self.model, FSDP):
                    self.model.clip_grad_norm_(self.max_grad_norm)
                    if isinstance(self.value_head, FSDP):
                        self.value_head.clip_grad_norm_(self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.value_head.parameters()), 
                        self.max_grad_norm
                    )
                
                self.optimizer.step()

        return {
            "loss": total_loss.item() if torch.isfinite(total_loss) else float('inf'),
            "policy_loss": policy_loss.item() if torch.isfinite(policy_loss) else float('inf'),
            "value_loss": value_loss.item() if torch.isfinite(value_loss) else float('inf'),
            "entropy": entropy.item() if torch.isfinite(entropy) else 0.0,
            "mean_return": returns.mean().item() if torch.isfinite(returns).all() else 0.0
        }

    def update_old_policy(self):
        """Update old policy with FSDP state dict handling"""
        if isinstance(self.model, FSDP):
            with FSDP.state_dict_type(
                self.model, 
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                model_state = self.model.state_dict()
            
            with FSDP.state_dict_type(
                self.old_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                self.old_model.load_state_dict(model_state)
        else:
            self.old_model.load_state_dict(self.model.state_dict())

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


# Model + tokenizer setup with FSDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

def setup_training_fsdp(rank, world_size):
    """Setup training with FSDP support"""
    
    # Initialize process group
    if world_size > 1:
        # Check if already initialized (when using torchrun)
        if not dist.is_initialized():
            # Manual initialization
            import socket
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    device = torch.device(f"cuda:{rank}")
    
    base = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    
    # Fix padding issue
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Load model (will be wrapped with FSDP)
    model = AutoModelForCausalLM.from_pretrained(
        base, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    # Create value head
    value_head = ValueHead(model.config.hidden_size)
    
    # Setup auto wrap policy for transformer layers
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen3DecoderLayer,  # For Qwen models
            GPT2Block,  # For GPT2 models
            LlamaDecoderLayer,  # For Llama models
            # Add other layer types as needed
        },
    )
    
    # Mixed precision config
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        reduce_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        buffer_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=rank,
        use_orig_params=True,
    )
    
    # Wrap value head with FSDP
    value_head = FSDP(
        value_head,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank,
        use_orig_params=True,
    )
    
    # Create optimizer
    optimizer = AdamW(
        list(model.parameters()) + list(value_head.parameters()), 
        lr=1e-5,
        weight_decay=0.01
    )
    
    # Load reward model
    reward_model = HFRewardModel().to(device)
    
    # Create trainer first (without old_model)
    trainer = PPOTrainerFSDP(
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
        max_grad_norm=0.5,
        rank=rank,
        world_size=world_size,
        use_mixed_precision=True
    )
    
    # SOLUTION 1: Create old model separately and assign it
    # Get state dict from the current model
    with FSDP.state_dict_type(
        model, 
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        model_state = model.state_dict()
    
    # Create old model
    old_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    # Load the state dict into old model before wrapping
    old_model.load_state_dict(model_state, strict=False)
    
    # Now wrap old model with FSDP
    old_model = FSDP(
        old_model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank,
        use_orig_params=False,
    )
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad = False
    
    # Assign old model to trainer
    trainer.old_model = old_model
    
    return trainer, tokenizer

from datasets import load_dataset
from random import choices

def generate_and_train_fsdp(prompts, trainer, tokenizer, max_new_tokens=32):
    """Generate responses and train with PPO (FSDP version)"""
    
    device = torch.device(f"cuda:{trainer.rank}")
    
    # Tokenize input prompts
    encodings = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=256
    )
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    prompt_lens = attention_mask.sum(dim=1).tolist()

    # SOLUTION 1: Use FSDP summon_full_params context for generation
    # This temporarily gathers all sharded parameters for generation
    with torch.no_grad():
        # Use summon_full_params to gather all parameters for generation
        with FSDP.summon_full_params(trainer.model, writeback=False):
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

    # Combine prompt + generation
    full_input_ids = torch.cat([input_ids, generated_ids], dim=1)
    full_attention_mask = torch.cat([
        attention_mask,
        torch.ones_like(generated_ids)
    ], dim=1)

    # Create masks
    actions = full_input_ids
    B, T = full_input_ids.shape
    prompt_mask = torch.zeros((B, T), dtype=torch.float32).to(device)
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
        if trainer.rank == 0:
            print(f"Training error: {e}")
        return {"loss": float('inf'), "policy_loss": float('inf'), "value_loss": float('inf'), "entropy": 0.0, "mean_return": 0.0}

def cleanup():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

# Main training function for single process
def train_single_gpu(rank, world_size):
    """Training function for each GPU"""
    
    # Setup
    trainer, tokenizer = setup_training_fsdp(rank, world_size)
    
    # Load dataset
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        prompt_texts = [
            example["text"] for example in dataset
            if example.get("text") and len(example["text"]) < 200
        ][:1000]
    except:
        if rank == 0:
            print("Loading fallback prompts")
        prompt_texts = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a short story about a robot.",
            "How do you make chocolate chip cookies?",
            "What causes the seasons to change?"
        ] * 200

    if rank == 0:
        print(f"Loaded {len(prompt_texts)} prompts")
    
    num_steps = 100
    batch_size = 2
    max_new_tokens = 32

    for step in range(num_steps):
        # Sample prompts
        prompts_batch = choices(prompt_texts, k=batch_size)

        # Run PPO update
        metrics = generate_and_train_fsdp(
            prompts=prompts_batch,
            trainer=trainer,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens
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
    
    # Cleanup
    cleanup()

# Main entry point
if __name__ == "__main__":
    import torch.multiprocessing as mp
    
    # Check if running with torchrun or single GPU
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running with torchrun - environment variables are already set
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # Set device based on local rank
        torch.cuda.set_device(local_rank)
        
        # Initialize process group (torchrun handles most of this)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        
        train_single_gpu(rank, world_size)
    else:
        # Single GPU or manual multiprocessing
        world_size = torch.cuda.device_count()
        if world_size > 1:
            # Set a default master port if not specified
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
            
            # Multi-GPU with spawn
            mp.spawn(train_single_gpu, args=(world_size,), nprocs=world_size, join=True)
        else:
            # Single GPU
            train_single_gpu(0, 1)