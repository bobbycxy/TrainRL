import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import numpy as np

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)
        # Proper initialization for value head
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)
    
    def forward(self, hidden_state):
        return self.value(hidden_state).squeeze(-1) # [B, T]
    
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

        self.old_model = deepcopy(model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

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
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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
                    print(f"Warning: Non-finite loss detected: {total_loss.item()}")
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
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


# Model + tokenizer setup
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

def setup_training():
    base = "Qwen/Qwen3-1.7B"  # Alternative model, or use your original "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    
    # Fix padding issue mentioned in the error
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # For generation

    model = AutoModelForCausalLM.from_pretrained(
        base, 
        torch_dtype=torch.float32
    ).cuda()
    
    value_head = ValueHead(model.config.hidden_size).cuda()

    # Use lower learning rate for stability
    optimizer = AdamW(
        list(model.parameters()) + list(value_head.parameters()), 
        lr=1e-5,  # Much lower learning rate
        weight_decay=0.01
    )
    
    reward_model = HFRewardModel().cuda()

    trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        value_head=value_head,
        optimizer=optimizer,
        reward_model=reward_model,
        clip_eps=0.2,
        c1=0.5,
        c2=0.01,
        ppo_epochs=2,  # Reduce epochs for stability
        batch_size=2,  # Smaller batch size
        max_grad_norm=0.5  # More aggressive gradient clipping
    )
    
    return trainer, tokenizer

from datasets import load_dataset
from random import choices

def generate_and_train(prompts, trainer, tokenizer, max_new_tokens=32):  # Reduced tokens
    """Generate responses and train with PPO"""
    
    # Tokenize input prompts
    encodings = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=256  # Limit input length
    )
    input_ids = encodings.input_ids.cuda()
    attention_mask = encodings.attention_mask.cuda()
    prompt_lens = attention_mask.sum(dim=1).tolist()

    # Generate responses
    with torch.no_grad():
        generation = trainer.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.7,  # Lower temperature for stability
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
    prompt_mask = torch.zeros((B, T), dtype=torch.float32).cuda()
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
        print(f"Training error: {e}")
        return {"loss": float('inf'), "policy_loss": float('inf'), "value_loss": float('inf'), "entropy": 0.0, "mean_return": 0.0}

# Main training loop
if __name__ == "__main__":
    trainer, tokenizer = setup_training()
    
    # Load dataset
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        prompt_texts = [
            example["text"] for example in dataset
            if example.get("text") and len(example["text"]) < 200  # Shorter prompts
        ][:1000]  # Limit dataset size
    except:
        # Fallback simple prompts if dataset loading fails
        print("LOADING FALLBACK")
        prompt_texts = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a short story about a robot.",
            "How do you make chocolate chip cookies?",
            "What causes the seasons to change?"
        ] * 200

    print(f"Loaded {len(prompt_texts)} prompts")
    
    num_steps = 100  # Reduced for testing
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
            max_new_tokens=max_new_tokens
        )

        print(f"[Step {step}] Loss: {metrics['loss']:.2f}, Policy: {metrics['policy_loss']:.2f}, "
              f"Value: {metrics['value_loss']:.2f}, Entropy: {metrics['entropy']:.4f}, Return: {metrics['mean_return']:.2f}")

        # Update old policy less frequently for stability
        if step % 8 == 0 and step > 0:
            trainer.update_old_policy()
            print("Updated old policy")
            
        # Early stopping if loss explodes
        if metrics['loss'] > 1000:
            print("Loss too high, stopping training")
            break