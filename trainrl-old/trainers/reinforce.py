import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from .base import BaseRLTrainer

logger = logging.getLogger(__name__)


def compute_advantages(rewards, values, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation (GAE)"""
    advantages = []
    returns = []
    
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]
        
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])
    
    return torch.tensor(advantages), torch.tensor(returns)


class REINFORCETrainer(BaseRLTrainer):
    """REINFORCE trainer for FSDP models with optional baseline"""
    
    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)
        
        # Optional: Create a simple value function as baseline (reduces variance)
        # In practice, you might want a separate value network
        self.use_baseline = config.baseline_coef > 0
        if self.use_baseline:
            self.baseline_history = []  # Simple moving average baseline
            self.baseline_window = 100
    
    def get_baseline(self, rewards):
        """Get baseline value to reduce variance (simple moving average)"""
        if not self.use_baseline:
            return 0.0
        
        # Add current rewards to history
        self.baseline_history.extend(rewards.tolist())
        
        # Keep only recent rewards
        if len(self.baseline_history) > self.baseline_window:
            self.baseline_history = self.baseline_history[-self.baseline_window:]
        
        # Return mean as baseline
        return np.mean(self.baseline_history) if self.baseline_history else 0.0
    
    def train_step(self, batch):
        """Single REINFORCE training step"""
        if dist.get_rank() == 0:
            print("Starting REINFORCE train_step...")
            
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
        
        # Get baseline to reduce variance
        baseline = self.get_baseline(rewards)
        advantages = rewards - baseline
        
        # Compute log probabilities using base class method
        prompt_length = input_ids.size(1)
        generated_log_probs_list = self.compute_log_probabilities(
            generated_ids, generated_text_ids, prompt_length
        )
        
        if not generated_log_probs_list:
            return {"loss": torch.tensor(0.0), "reward": rewards.mean().item()}
        
        # For REINFORCE, we sum the log probabilities for each sequence
        log_probs_tensor = []
        for seq_log_probs in generated_log_probs_list:
            # Sum log probs for the sequence (REINFORCE uses sum, not mean)
            total_log_prob = seq_log_probs.sum()
            log_probs_tensor.append(total_log_prob)
        
        log_probs_tensor = torch.stack(log_probs_tensor)
        
        # REINFORCE loss: -log_prob * advantage
        # Negative because we want to maximize reward (minimize negative reward)
        policy_loss = -(log_probs_tensor * advantages).mean()
        
        # Optional: Add entropy bonus to encourage exploration
        # For simplicity, we'll skip this in the basic REINFORCE implementation
        
        total_loss = policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        
        # Get common metrics template
        metrics = self.get_training_metrics_template(questions, generated_texts, rewards)
        
        # Add REINFORCE-specific metrics
        metrics.update({
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "reward": rewards.mean().item(),
            "baseline": baseline,
            "advantage": advantages.mean().item(),
        })
        
        return metrics