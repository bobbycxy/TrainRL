import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from trainer.utils import get_log_probs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import torch.distributed as dist
    
class PPOTrainer:
    def __init__(
        self, cfg, model, old_model, value_head, optimizer, reward_model,
        # gamma=0.99, lam=0.95, clip_eps=0.2, c1=0.5, c2=0.01,
        # ppo_epochs=4, batch_size=4, max_grad_norm=1.0, grad_accum_steps=4, 
        device="cuda"
    ):
        self.model = model
        self.old_model = old_model
        self.value_head = value_head
        self.optimizer = optimizer
        self.reward_model = reward_model
        self.gamma = cfg.ppo.gamma
        self.lam = cfg.ppo.lam
        self.clip_eps = cfg.ppo.clip_eps
        self.c1 = cfg.ppo.c1
        self.c2 = cfg.ppo.c2
        self.ppo_epochs = cfg.ppo.ppo_epochs
        self.batch_size = cfg.ppo.ppo_batch_size
        self.max_grad_norm = cfg.ppo.max_grad_norm
        self.grad_accum_steps = cfg.ppo.grad_accum_steps
        self.device = device

    def compute_gae_masked(self, rewards, values, attention_mask, prompt_mask):
        """
        rewards, values: [B, T]
        attention_mask: [B, T] (1 for real tokens incl. prompt, 0 for pad)
        prompt_mask:    [B, T] (1 for generated tokens, 0 for prompt)
        """
        # 1) Valid = generated & not padded
        valid = (attention_mask > 0).float() * (prompt_mask > 0).float()          # [B, T]

        # 2) Continuation mask: 1 if next step is also a valid generated step
        valid_next = torch.nn.functional.pad(valid[:, 1:], (0, 1))                # shift left, pad 0 at end
        cont = valid * valid_next                                                 # [B, T]

        # 3) Next values (0 past the end)
        values_next = torch.nn.functional.pad(values[:, 1:], (0, 1))              # [B, T]

        # 4) TD residual
        delta = rewards + self.gamma * values_next * cont - values                # [B, T]

        # 5) Backward GAE over time (masked)
        B, T = rewards.shape
        adv = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        for t in range(T - 1, -1, -1):
            gae = delta[:, t] + self.gamma * self.lam * cont[:, t] * gae
            adv[:, t] = gae * valid[:, t]     # zero out prompts/pads early

        # 6) Returns
        ret = adv + values

        # 7) Normalize advantages over generated tokens only
        denom = valid.sum()
        if denom > 1:  # Need at least 2 samples for meaningful normalization
            # Compute mean and variance only over valid (generated) tokens
            valid_adv = adv * valid
            mean = valid_adv.sum() / denom
            var = ((adv - mean) ** 2 * valid).sum() / denom
            std = (var.clamp_min(1e-12)).sqrt()
            adv = (adv - mean) * valid / (std + 1e-8)
        elif denom == 1:
            # Single generated token - just center it
            valid_adv = adv * valid
            mean = valid_adv.sum()
            adv = (adv - mean) * valid

        return adv.detach(), ret.detach()

    def train(self, actions, attention_mask, prompt_mask, rewards, old_log_probs, old_values, scaler=None):
        """
        PPO training loop with gradient accumulation and optional AMP.
        - scaler: optional torch.cuda.amp.GradScaler for mixed precision training
        """

        total_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        batch_count = 0

        for epoch in range(self.ppo_epochs):
            dataset = TensorDataset(actions, attention_mask, prompt_mask, rewards, old_log_probs, old_values)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(dataloader):
                actions_b, attn_mask_b, prompt_mask_b, rewards_b, old_log_probs_b, old_values_b = batch

                with torch.amp.autocast(enabled=(scaler is not None), device_type=self.device):
                    # Forward pass
                    logits, _ = self.model(actions_b, attn_mask=attn_mask_b)
                    log_probs = get_log_probs(logits, actions_b)

                    # Value predictions
                    hidden_states = self.model.generate_hidden_states(actions_b, attn_mask=attn_mask_b)
                    values, _ = self.value_head(hidden_states)
                    values = values.squeeze(-1)

                    # Compute GAE
                    advantages, returns = self.compute_gae_masked(
                        rewards=rewards_b,
                        values=values,
                        attention_mask=attn_mask_b,
                        prompt_mask=prompt_mask_b,
                    )

                    # Policy loss
                    ratios = torch.exp(log_probs - old_log_probs_b)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2)
                    policy_loss = (policy_loss * prompt_mask_b).sum() / (prompt_mask_b.sum() + 1e-8)

                    # Value loss
                    value_pred_clipped = old_values_b + torch.clamp(values - old_values_b, -self.clip_eps, self.clip_eps)
                    value_loss1 = (values - returns) ** 2
                    value_loss2 = (value_pred_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                    value_loss = (value_loss * prompt_mask_b).sum() / (prompt_mask_b.sum() + 1e-8)

                    # Entropy
                    entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1)
                    entropy = (entropy * prompt_mask_b).sum() / (prompt_mask_b.sum() + 1e-8)

                    # PPO total loss
                    total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

                if not torch.isfinite(total_loss):
                    print(f"Warning: Non-finite loss detected: {total_loss.item()}")
                    continue

                scaled_loss = total_loss / self.grad_accum_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                total_loss_sum += total_loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.item()
                batch_count += 1

                if (i + 1) % self.grad_accum_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)  # unscale before clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.value_head.parameters()),
                        self.max_grad_norm,
                    )
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)  # reset grads for next accumulation window

                    del logits, log_probs, values, advantages, returns, hidden_states
                    torch.cuda.empty_cache()

        if batch_count > 0:
            return {
                "loss": total_loss_sum / batch_count,
                "policy_loss": policy_loss_sum / batch_count,
                "value_loss": value_loss_sum / batch_count,
                "entropy": entropy_sum / batch_count,
                "mean_return": returns.mean().item() if torch.isfinite(returns).all() else 0.0,
                "mean_advantage": advantages.mean().item() if torch.isfinite(advantages).all() else 0.0,
                "generated_tokens": prompt_mask.sum().item(),
                "batches_processed": batch_count,
            }
        else:
            return {
                "loss": float("inf"),
                "policy_loss": float("inf"),
                "value_loss": float("inf"),
                "entropy": 0.0,
                "mean_return": 0.0,
                "mean_advantage": 0.0,
                "generated_tokens": 0,
                "batches_processed": 0,
            }

    def update_old_policy(self):
        """Update old policy for next iteration"""
        # Fast path: copy LOCAL shards between identically wrapped FSDP models
        if isinstance(self.model, FSDP) and isinstance(self.old_model, FSDP):
            with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT), \
                FSDP.state_dict_type(self.old_model, StateDictType.LOCAL_STATE_DICT):
                local_sd = self.model.state_dict()
                self.old_model.load_state_dict(local_sd, strict=False)
        else:
            # Single-GPU / non-FSDP fallback
            self.old_model.load_state_dict(self.model.state_dict(), strict=False)

        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False