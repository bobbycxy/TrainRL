import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)
        # Proper initialization for value head
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)
    
    def forward(self, hidden_state):
        return self.value(hidden_state).squeeze(-1) # [B, T]
    
import torch
from torch.nn.utils.rnn import pad_sequence

class RolloutBuffer:
    def __init__(self):
        self.input_ids = []
        self.attn_masks = []
        self.prompt_masks = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []
        self.old_values = []

    def add(self, input_ids, attn_mask, prompt_mask, actions, rewards, log_probs, values):
        self.input_ids.append(input_ids)
        self.attn_masks.append(attn_mask)
        self.prompt_masks.append(prompt_mask)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.old_log_probs.append(log_probs)
        self.old_values.append(values)

    def get(self):
        # Pad sequences to the same length
        def pad(tensors):
            return pad_sequence(tensors, batch_first=True, padding_value=0)

        return (
            pad(self.input_ids),
            pad(self.attn_masks),
            pad(self.prompt_masks),
            pad(self.actions),
            pad(self.rewards),
            pad(self.old_log_probs),
            pad(self.old_values),
        )

    def clear(self):
        self.__init__()


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
    
    def train(self, input_ids, attention_mask, prompt_mask, actions, rewards, old_log_probs, old_values):
        """
        input_ids: [B, T] (prompt + completion)
        attention_mask: [B, T] 
        prompt_mask: [B, T] where 1.0 = generated token, 0.0 = prompt token
        actions: same as input_ids (for autoregressive LM)
        """
        device = input_ids.device

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