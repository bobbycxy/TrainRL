import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, MixedPrecision
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from functools import partial

from .core import RewardFunction

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_length: int = 512
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    gradient_clip: float = 1.0
    baseline_decay: float = 0.99
    temperature: float = 1.0
    # FSDP specific configs
    sharding_strategy: str = "FULL_SHARD"  # Options: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    mixed_precision: bool = True
    cpu_offload: bool = False
    activation_checkpointing: bool = False

class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

class REINFORCETrainer:
    def __init__(self, model_name: str, reward_function: RewardFunction, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.reward_function = reward_function

        # Initialize distributed training if available
        self.is_distributed = dist.is_available() and dist.is_initialized()
        
        if self.is_distributed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
        else:
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model with FSDP if distributed
        if self.is_distributed:
            self._init_fsdp_model(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)

        # Initialize optimizer after model wrapping
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.baseline = 0.0

    def _get_fsdp_wrap_policy(self):
        """Get the FSDP auto wrap policy for transformer models"""
        # This will automatically wrap transformer blocks based on the model architecture
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
        
        # Define transformer layer classes that should be wrapped
        transformer_layer_cls = {
            GPT2Block,  # GPT-2
            LlamaDecoderLayer,  # Llama
            MistralDecoderLayer,  # Mistral
        }
        
        # Add more model types as needed
        try:
            from transformers.models.phi.modeling_phi import PhiDecoderLayer
            transformer_layer_cls.add(PhiDecoderLayer)
        except ImportError:
            pass
            
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )

    def _get_mixed_precision_policy(self):
        """Get mixed precision policy for FSDP"""
        if not self.config.mixed_precision:
            return None
            
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    def _get_sharding_strategy(self):
        """Get FSDP sharding strategy"""
        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        return strategy_map.get(self.config.sharding_strategy, ShardingStrategy.FULL_SHARD)

    def _init_fsdp_model(self, model_name: str):
        """Initialize model with FSDP"""
        # Load model on CPU first to avoid OOM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
        )
        
        # Store the original model name for potential fallback
        if hasattr(model.config, 'name_or_path'):
            model.config.name_or_path = model_name
        else:
            model.config._name_or_path = model_name
        
        # Use a more conservative FSDP configuration to avoid sharding issues
        fsdp_config = {
            "sharding_strategy": self._get_sharding_strategy(),
            "device_id": self.local_rank,
            "sync_module_states": True,
            "use_orig_params": True,  # Use original parameters for better compatibility
        }
        
        # Only add advanced features if using FULL_SHARD or SHARD_GRAD_OP
        if self.config.sharding_strategy in ["FULL_SHARD", "SHARD_GRAD_OP"]:
            fsdp_config.update({
                "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
                "auto_wrap_policy": self._get_fsdp_wrap_policy(),
            })
            
            # Add mixed precision only for FULL_SHARD
            if self.config.sharding_strategy == "FULL_SHARD" and self.config.mixed_precision:
                fsdp_config["mixed_precision"] = self._get_mixed_precision_policy()
        
        # Add CPU offloading if enabled (only for FULL_SHARD)
        if self.config.cpu_offload and self.config.sharding_strategy == "FULL_SHARD":
            from torch.distributed.fsdp import CPUOffload
            fsdp_config["cpu_offload"] = CPUOffload(offload_params=True)
        
        # Wrap model with FSDP
        try:
            self.model = FSDP(model, **fsdp_config)
            if self.rank == 0:
                logger.info(f"FSDP model initialized successfully with strategy: {self.config.sharding_strategy}")
        except Exception as e:
            if self.rank == 0:
                logger.warning(f"FSDP initialization failed: {e}")
                logger.warning("Falling back to simpler FSDP configuration")
            
            # Fallback with minimal config
            simplified_config = {
                "sharding_strategy": ShardingStrategy.NO_SHARD,  # Most compatible
                "device_id": self.local_rank,
                "sync_module_states": True,
                "use_orig_params": True,
            }
            self.model = FSDP(model, **simplified_config)
            if self.rank == 0:
                logger.info("Using NO_SHARD fallback configuration")
        
        # Enable activation checkpointing if specified
        if self.config.activation_checkpointing:
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    CheckpointImpl,
                    apply_activation_checkpointing,
                )
                
                # More general check function for transformer layers
                def check_fn(submodule):
                    return any(
                        layer_type in str(type(submodule)).lower() 
                        for layer_type in ['block', 'layer', 'decoder']
                    )
                
                apply_activation_checkpointing(
                    self.model,
                    checkpoint_wrapper_fn=checkpoint_wrapper,
                    check_fn=check_fn,
                )
                if self.rank == 0:
                    logger.info("Activation checkpointing enabled")
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f"Failed to enable activation checkpointing: {e}")

    def _get_unwrapped_model(self):
        """Get the unwrapped model (useful for FSDP)"""
        if self.is_distributed and isinstance(self.model, FSDP):
            return self.model.module
        return self.model

    def generate_response(self, prompt: str, do_sample: bool = True) -> Tuple[str, torch.Tensor, torch.Tensor]:
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_tokens.shape[1]

        # For FSDP, we need to handle generation more carefully
        self.model.eval()
        
        if self.is_distributed and isinstance(self.model, FSDP):
            # Instead of summon_full_params, we'll use FSDP in eval mode with no_sync
            # This prevents the assertion error by not trying to unshard during generation
            with torch.no_grad():
                # Use the model directly - FSDP will handle parameter gathering internally
                # Set model to eval mode to avoid training-specific FSDP behaviors
                try:
                    outputs = self.model.generate(
                        prompt_tokens,
                        max_length=min(prompt_length + 200, self.config.max_length),
                        do_sample=do_sample,
                        temperature=self.config.temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True if do_sample else False,
                        # Add these parameters to help with FSDP generation
                        use_cache=True,
                        output_hidden_states=False,
                        output_attentions=False,
                    )
                    generated_tokens = outputs.sequences[0][prompt_length:]
                except Exception as e:
                    if self.rank == 0:
                        logger.warning(f"FSDP generation failed: {e}, falling back to simple generation")
                    
                    # Fallback: generate on rank 0 and broadcast
                    if self.rank == 0:
                        # Temporarily switch to NO_SHARD for generation
                        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                        
                        # Get full state dict on rank 0
                        fullstate_save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
                        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
                            state_dict = self.model.state_dict()
                        
                        # Create temporary model for generation
                        temp_model = AutoModelForCausalLM.from_pretrained(
                            self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else 'gpt2',
                            torch_dtype=self.model.dtype if hasattr(self.model, 'dtype') else torch.float32
                        )
                        temp_model.load_state_dict(state_dict)
                        temp_model.to(self.device)
                        temp_model.eval()
                        
                        outputs = temp_model.generate(
                            prompt_tokens,
                            max_length=min(prompt_length + 200, self.config.max_length),
                            do_sample=do_sample,
                            temperature=self.config.temperature,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=True if do_sample else False
                        )
                        generated_tokens = outputs.sequences[0][prompt_length:]
                        
                        # Clean up temporary model
                        del temp_model
                        torch.cuda.empty_cache()
                        
                        generated_tokens_list = generated_tokens.cpu().tolist()
                    else:
                        generated_tokens_list = None
                    
                    # Broadcast the result to all ranks
                    generated_tokens_list = [generated_tokens_list]
                    dist.broadcast_object_list(generated_tokens_list, src=0)
                    generated_tokens = torch.tensor(generated_tokens_list[0], device=self.device)
        else:
            # Standard generation for non-FSDP models
            with torch.no_grad():
                outputs = self.model.generate(
                    prompt_tokens,
                    max_length=min(prompt_length + 200, self.config.max_length),
                    do_sample=do_sample,
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True if do_sample else False
                )
                generated_tokens = outputs.sequences[0][prompt_length:]

        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if do_sample:
            full_tokens = torch.cat([prompt_tokens, generated_tokens.unsqueeze(0)], dim=1)
            log_probs = self._compute_log_probs(full_tokens, prompt_length)
        else:
            log_probs = torch.zeros(len(generated_tokens), device=self.device)

        return response_text, log_probs, generated_tokens

    def _compute_log_probs(self, tokens: torch.Tensor, prompt_length: int) -> torch.Tensor:
        self.model.train()  # Switch back to train mode for gradient computation
        
        if self.is_distributed and isinstance(self.model, FSDP):
            # For FSDP, we don't need summon_full_params during training forward pass
            # as FSDP handles parameter gathering automatically during forward
            outputs = self.model(tokens)
        else:
            outputs = self.model(tokens)
            
        logits = outputs.logits
        generated_logits = logits[0, prompt_length - 1:-1]
        generated_tokens = tokens[0, prompt_length:]
        log_probs = torch.log_softmax(generated_logits, dim=-1)
        return log_probs.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)

    def _aggregate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate metrics across all processes in distributed training"""
        if not self.is_distributed:
            return metrics
        
        aggregated_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            aggregated_metrics[key] = (tensor / self.world_size).item()
        
        return aggregated_metrics

    def _should_log(self) -> bool:
        """Check if current process should log (only rank 0 in distributed)"""
        return not self.is_distributed or self.rank == 0

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        self.model.train()
        batch_rewards = []
        batch_log_probs = []

        # Generate responses and compute rewards
        for prompt in prompts:
            response, log_probs, _ = self.generate_response(prompt, do_sample=True)
            reward = self.reward_function(prompt, response)
            batch_rewards.append(reward)
            batch_log_probs.append(log_probs)

        # Update baseline
        current_avg_reward = sum(batch_rewards) / len(batch_rewards)
        self.baseline = self.config.baseline_decay * self.baseline + (1 - self.config.baseline_decay) * current_avg_reward

        # Compute loss
        total_loss = 0
        advantages = []
        for reward, log_probs in zip(batch_rewards, batch_log_probs):
            advantage = reward - self.baseline
            advantages.append(advantage)
            if len(log_probs) > 0:  # Ensure we have generated tokens
                loss = -log_probs.sum() * advantage
                total_loss += loss

        if len(batch_rewards) > 0:
            total_loss = total_loss / len(batch_rewards)
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Backward pass with FSDP
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping with FSDP
        if self.is_distributed and isinstance(self.model, FSDP):
            self.model.clip_grad_norm_(self.config.gradient_clip)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        self.optimizer.step()

        # Prepare local metrics
        local_metrics = {
            "loss": total_loss.item(),
            "avg_reward": current_avg_reward,
            "baseline": self.baseline,
            "avg_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
            "batch_size": len(prompts)
        }

        # Aggregate metrics across all processes
        aggregated_metrics = self._aggregate_metrics(local_metrics)

        return aggregated_metrics

    def evaluate(self, eval_prompts: List[str]) -> Dict[str, float]:
        """Evaluate the model on evaluation prompts"""
        self.model.eval()
        eval_rewards = []
        
        with torch.no_grad():
            for prompt in eval_prompts:
                response, _, _ = self.generate_response(prompt, do_sample=False)
                reward = self.reward_function(prompt, response)
                eval_rewards.append(reward)
        
        local_eval_metrics = {
            "eval_avg_reward": sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0,
            "eval_samples": len(eval_rewards)
        }
        
        # Aggregate evaluation metrics
        aggregated_eval_metrics = self._aggregate_metrics(local_eval_metrics)
        
        return aggregated_eval_metrics

    def train(self, train_prompts: List[str], eval_prompts: Optional[List[str]] = None) -> Dict[str, List[float]]:
        history = {"loss": [], "avg_reward": [], "baseline": [], "eval_reward": [], "avg_advantage": []}

        # Create dataset and dataloader
        train_dataset = PromptDataset(train_prompts)
        
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=True
            )
            train_loader = DataLoader(
                train_dataset, 
                sampler=train_sampler, 
                batch_size=self.config.batch_size,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size,
                shuffle=True,
                pin_memory=True
            )

        total_steps = 0

        for epoch in range(self.config.num_epochs):
            if self.is_distributed:
                train_sampler.set_epoch(epoch)
            
            epoch_progress = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                disable=not self._should_log()
            )
            
            for step, batch_prompts in enumerate(epoch_progress):
                # Convert batch_prompts to list if it's a tensor
                if isinstance(batch_prompts, torch.Tensor):
                    batch_prompts = [train_prompts[i] for i in batch_prompts]
                
                metrics = self.train_step(batch_prompts)
                total_steps += 1

                # Update progress bar
                if self._should_log():
                    epoch_progress.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'reward': f"{metrics['avg_reward']:.4f}",
                        'baseline': f"{metrics['baseline']:.4f}"
                    })

                # Log metrics
                if total_steps % self.config.logging_steps == 0:
                    if self._should_log():
                        logger.info(f"Epoch {epoch+1}, Step {total_steps}: {metrics}")
                    
                    # Store history (only on rank 0 to avoid duplication)
                    if self._should_log():
                        history["loss"].append(metrics["loss"])
                        history["avg_reward"].append(metrics["avg_reward"])
                        history["baseline"].append(metrics["baseline"])
                        history["avg_advantage"].append(metrics["avg_advantage"])

                # Evaluation
                if eval_prompts and total_steps % (self.config.logging_steps * 2) == 0:
                    eval_metrics = self.evaluate(eval_prompts)
                    if self._should_log():
                        logger.info(f"Evaluation at step {total_steps}: {eval_metrics}")
                        history["eval_reward"].append(eval_metrics["eval_avg_reward"])

        # Final evaluation
        if eval_prompts:
            final_eval_metrics = self.evaluate(eval_prompts)
            if self._should_log():
                logger.info(f"Final evaluation: {final_eval_metrics}")
                history["eval_reward"].append(final_eval_metrics["eval_avg_reward"])

        return history

    def save_model(self, save_path: str):
        """Save the model with FSDP support (only on rank 0)"""
        if self._should_log():
            os.makedirs(save_path, exist_ok=True)
            
            if self.is_distributed and isinstance(self.model, FSDP):
                # FSDP model saving
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                
                # Configure how to save the state dict
                fullstate_save_policy = FullStateDictConfig(
                    offload_to_cpu=True, 
                    rank0_only=True
                )
                
                with FSDP.state_dict_type(
                    self.model, 
                    StateDictType.FULL_STATE_DICT, 
                    fullstate_save_policy
                ):
                    model_state_dict = self.model.state_dict()
                
                # Save using standard transformers method if on rank 0
                if self.rank == 0:
                    # Manually save the state dict and config
                    unwrapped_model = self._get_unwrapped_model()
                    
                    # Save model config and weights
                    unwrapped_model.config.save_pretrained(save_path)
                    torch.save(model_state_dict, os.path.join(save_path, "pytorch_model.bin"))
                    
                    # Save tokenizer
                    self.tokenizer.save_pretrained(save_path)
                    
                    # Save additional training state
                    torch.save({
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'baseline': self.baseline,
                        'config': self.config
                    }, os.path.join(save_path, "training_state.pt"))
                    
                    logger.info(f"FSDP model saved to {save_path}")
            else:
                # Standard model saving
                unwrapped_model = self._get_unwrapped_model()
                unwrapped_model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                
                # Save additional training state
                torch.save({
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'baseline': self.baseline,
                    'config': self.config
                }, os.path.join(save_path, "training_state.pt"))
                
                logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """Load the model with FSDP support"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        if self.is_distributed and isinstance(self.model, FSDP):
            # FSDP model loading
            model_path = os.path.join(load_path, "pytorch_model.bin")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                
                # Load state dict into FSDP model
                self.model.load_state_dict(state_dict)
                logger.info(f"FSDP model weights loaded from {model_path}")
        else:
            # Standard model loading
            unwrapped_model = self._get_unwrapped_model()
            unwrapped_model.from_pretrained(load_path)
        
        # Load additional training state
        state_path = os.path.join(load_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.baseline = state['baseline']
            logger.info(f"Training state loaded from {state_path}")
        
        logger.info(f"Model loaded from {load_path}")