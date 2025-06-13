import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def generate_with_model(model, tokenizer, input_ids, attention_mask, max_new_tokens=150):
    """Generate text using the model (manual implementation for FSDP compatibility)"""
    if dist.get_rank() == 0:
        print(f"Generate: input_ids shape {input_ids.shape}, max_new_tokens {max_new_tokens}")
    
    model.eval()
    generated_ids = input_ids.clone()
    batch_size = input_ids.size(0)
    
    # Track which sequences are finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            if dist.get_rank() == 0 and step % 5 == 0:
                print(f"Generation step {step}/{max_new_tokens}")
                
            # Skip if all sequences are finished
            if finished.all():
                break
                
            # Get current attention mask
            current_attention_mask = torch.ones_like(generated_ids)
            
            try:
                outputs = model(generated_ids, attention_mask=current_attention_mask)
                logits = outputs.logits
                
                # Get next token probabilities
                next_token_logits = logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token (you could also use greedy or beam search)
                next_token = torch.multinomial(next_token_probs, 1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Check for EOS tokens in the batch
                if tokenizer.eos_token_id is not None:
                    # Mark sequences as finished if they generate EOS token
                    eos_generated = (next_token.squeeze(-1) == tokenizer.eos_token_id)
                    finished = finished | eos_generated
                    
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"Error during generation step {step}: {e}")
                break
    
    if dist.get_rank() == 0:
        print(f"Generation completed. Final shape: {generated_ids.shape}")
    
    return generated_ids


class RewardFunction:
    """Rule-based reward function for mathematical reasoning"""
    
    def __init__(self):
        self.correct_answer_reward = 10.0
        self.partial_credit_reward = 2.0
        self.format_penalty = -1.0
        self.length_penalty_factor = -0.01
    
    def extract_answer_from_generation(self, text):
        """Extract numerical answer from generated text"""
        import re
        
        # Look for patterns like "The answer is X" or "Answer: X" or just numbers
        patterns = [
            r"(?:the answer is|answer is|answer:|equals?|=)\s*(-?\d+(?:\.\d+)?)",
            r"####\s*(-?\d+(?:\.\d+)?)",  # GSM8K format
            r"(-?\d+(?:\.\d+)?)(?:\s*$|\s*\.?\s*$)",  # Number at end
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    continue
        
        # Fallback: extract all numbers and take the last one
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def calculate_reward(self, generated_text, correct_answer, question=""):
        """Calculate reward based on generated text and correct answer"""
        reward = 0.0
        
        # Length penalty (encourage conciseness)
        length_penalty = len(generated_text.split()) * self.length_penalty_factor
        reward += length_penalty
        
        # Extract predicted answer
        predicted_answer = self.extract_answer_from_generation(generated_text)
        
        if predicted_answer is None:
            # No numerical answer found - penalty
            reward += self.format_penalty
            return reward
        
        # Check if answer is correct
        if abs(predicted_answer - correct_answer) < 1e-6:
            reward += self.correct_answer_reward
        elif abs(predicted_answer - correct_answer) < abs(correct_answer * 0.1):
            # Within 10% - partial credit
            reward += self.partial_credit_reward
        
        # Bonus for showing work (contains mathematical operations)
        if any(op in generated_text for op in ['+', '-', '*', '/', '×', '÷', '=']):
            reward += 1.0
        
        return reward


class BaseRLTrainer:
    """Base class for RL trainers"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = RewardFunction()
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate
        )
    
    def generate_completions(self, input_ids, attention_mask):
        """Generate completions for given inputs"""
        if dist.get_rank() == 0:
            print("Starting generation...")
            
        generated_ids = generate_with_model(
            self.model, 
            self.tokenizer, 
            input_ids, 
            attention_mask,
            max_new_tokens=self.config.generation_length
        )
        
        if dist.get_rank() == 0:
            print("Generation completed!")
        
        return generated_ids
    
    def calculate_rewards(self, generated_ids, input_ids, questions, correct_answers):
        """Calculate rewards for generated completions"""
        prompt_length = input_ids.size(1)
        generated_text_ids = generated_ids[:, prompt_length:]
        batch_size = generated_text_ids.size(0)
        
        rewards = []
        generated_texts = []
        
        for i in range(batch_size):
            generated_text = self.tokenizer.decode(
                generated_text_ids[i], 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
            
            reward = self.reward_fn.calculate_reward(
                generated_text, 
                correct_answers[i].item(),
                questions[i]
            )
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
        
        return rewards, generated_texts, generated_text_ids
    
    def compute_log_probabilities(self, generated_ids, generated_text_ids, prompt_length):
        """Compute log probabilities for generated tokens"""
        # Get model outputs for the generated sequence
        self.model.train()
        outputs = self.model(generated_ids, attention_mask=torch.ones_like(generated_ids))
        
        # Calculate log probabilities
        logits = outputs.logits[:, prompt_length-1:-1, :]  # Shift for next token prediction
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log probabilities of generated tokens
        generated_log_probs = []
        batch_size = generated_text_ids.size(0)
        
        for i in range(batch_size):
            seq_log_probs = []
            for j, token_id in enumerate(generated_text_ids[i]):
                if j < log_probs.size(1) and token_id < log_probs.size(2):
                    token_log_prob = log_probs[i, j, token_id]
                    seq_log_probs.append(token_log_prob)
            
            if seq_log_probs:
                # This will be handled differently by subclasses
                total_log_prob = torch.stack(seq_log_probs)
                generated_log_probs.append(total_log_prob)
        
        return generated_log_probs
    
    def train_step(self, batch):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train_step method")
    
    def get_training_metrics_template(self, questions, generated_texts, rewards):
        """Common metrics for training"""
        return {
            "generated_samples": list(zip(questions[:2], generated_texts[:2], rewards[:2].tolist()))
        }