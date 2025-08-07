import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from .base import BaseRLTrainer

logger = logging.getLogger(__name__)


class PPOTrainer(BaseRLTrainer):
    """PPO trainer for FSDP models"""
    
    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)
        
        # PPO-specific hyperparameters
        self.clip_epsilon = getattr(config, 'clip_epsilon', 0.2)
        self.value_loss_coef = getattr(config, 'value_loss_coef', 0.5)
        self.entropy_coef = getattr(config, 'entropy_coef', 0.01)
        
        # For full PPO implementation, you might want to store old policies
        # This is a simplified version that uses policy gradient with clipping
        
    def train_step(self, batch):
        """Single PPO training step"""
        if dist.get_rank() == 0:
            print("Starting PPO train_step...")
            
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        questions = batch["question"]
        correct_answers = batch["correct_answer"]
        
        batch_size = input_ids.size(0)
        
        if dist.get_rank() == 0:
            print(f"Batch size: {batch_size}, Input shape: {input_ids.shape}")
        
        # Generate completions using base class method
        generated_ids = self.generate_completions(input_ids, attention_mask)
        
        # Calculate rewards using base class method
        rewards, generated_texts, generated_text_ids = self.calculate_rewards(
            generated_ids, input_ids, questions, correct_answers
        )
        
        # Compute log probabilities using base class method
        prompt_length = input_ids.size(1)
        generated_log_probs_list = self.compute_log_probabilities(
            generated_ids, generated_text_ids, prompt_length
        )
        
        if not generated_log_probs_list:
            return {"loss": torch.tensor(0.0), "reward": rewards.mean().item()}
        
        # For PPO, we typically use average log probabilities
        log_probs_tensor = []
        for seq_log_probs in generated_log_probs_list:
            # Use mean instead of sum for PPO (more stable)
            avg_log_prob = seq_log_probs.mean()
            log_probs_tensor.append(avg_log_prob)
        
        log_probs_tensor = torch.stack(log_probs_tensor)
        
        # Simple policy gradient loss (REINFORCE-style)
        # For full PPO, you'd need to store old probabilities and compute ratios
        # This is a simplified version that demonstrates the structure
        policy_loss = -(log_probs_tensor * rewards).mean()
        
        # Optional: Add entropy loss to encourage exploration
        # For a full implementation, you'd compute actual entropy from the distribution
        entropy_loss = 0.0  # Placeholder
        
        # Optional: Add value loss if using a value function
        # For this simplified version, we skip the value function
        value_loss = 0.0  # Placeholder
        
        # Total loss (PPO combines policy, value, and entropy losses)
        total_loss = (
            policy_loss + 
            self.value_loss_coef * value_loss - 
            self.entropy_coef * entropy_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        
        # Get common metrics template
        metrics = self.get_training_metrics_template(questions, generated_texts, rewards)
        
        # Add PPO-specific metrics
        metrics.update({
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "reward": rewards.mean().item(),
        })
        
        return metrics
    
    def compute_ppo_loss(self, log_probs, old_log_probs, advantages):
        """
        Compute PPO clipped loss
        Note: This is a placeholder for the full PPO implementation
        """
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Compute clipped loss
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # PPO loss is the minimum of the clipped and unclipped objective
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        
        policy_loss = -torch.min(loss1, loss2).mean()
        
        return policy_loss