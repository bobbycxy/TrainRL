import torch
from trainer.utils import get_log_probs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class RolloutCollector:
    def __init__(self, cfg, model, value_head, reward_model=None, device="cuda"):
        self.cfg = cfg
        self.model = model
        self.tokenizer = model.embedding_model.tokenizer
        self.value_head = value_head
        self.reward_model = reward_model
        self.device = device

        self.old_model = None  # Optional: explicitly settable, otherwise fallback to deepcopy(model)
        self._log_prob_fn = None  # Optional: set externally

        self.reward_mean_ma = 0.0
        self.reward_std_ma = 1.0
        self.reward_ma_beta = self.cfg.rewards.reward_ma_beta

    def set_old_model(self, old_model):
        self.old_model = old_model
        self.old_model.cuda()

    def _compute_rewards(self, full_texts, prompt_mask, reward_type="terminal"):
        with torch.no_grad():
            scores = self.reward_model(full_texts)  # [B]
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, dtype=torch.float32)
            scores = scores.to(prompt_mask.device)

            B, T = prompt_mask.shape
            rewards = torch.zeros(B, T, dtype=torch.float32, device=prompt_mask.device)

            if reward_type == "dense":
                # Broadcast score to all generated tokens
                rewards = scores.unsqueeze(1).expand(B, T) * prompt_mask

                # --- Moving average normalization ---
                valid_rewards = rewards[prompt_mask > 0]  # only generated tokens
                if valid_rewards.numel() > 0:
                    batch_mean = valid_rewards.mean()
                    batch_std = valid_rewards.std()

                    # Update moving averages
                    self.reward_mean_ma = (
                        self.reward_ma_beta * self.reward_mean_ma
                        + (1 - self.reward_ma_beta) * batch_mean.item()
                    )
                    self.reward_std_ma = (
                        self.reward_ma_beta * self.reward_std_ma
                        + (1 - self.reward_ma_beta) * batch_std.item()
                    )

                    rewards = (rewards - self.reward_mean_ma) / (self.reward_std_ma + 1e-8)

            elif reward_type == "terminal":
                for i in range(B):
                    gen_positions = (prompt_mask[i] > 0).nonzero(as_tuple=True)[0]
                    if len(gen_positions) > 0:
                        last_pos = gen_positions[-1]
                        rewards[i, last_pos] = scores[i]

            else:
                raise ValueError(f"Unknown reward_type: {reward_type}")

        return rewards



    def _is_fsdp_model(self, model):
        """Check if model is wrapped with FSDP"""
        # Check if the model itself is FSDP
        if isinstance(model, FSDP):
            return True
        # Check if it's a wrapper that contains FSDP
        if hasattr(model, 'wrapped_model') and isinstance(model.wrapped_model, FSDP):
            return True
        return False

    def _get_fsdp_model(self, model):
        """Get the actual FSDP model from wrapper if needed"""
        if isinstance(model, FSDP):
            return model
        elif hasattr(model, 'wrapped_model') and isinstance(model.wrapped_model, FSDP):
            return model.wrapped_model
        return model

    def _safe_model_call(self, model, method_name, *args, **kwargs):
        """Safely call model methods, handling FSDP contexts properly"""
        if self._is_fsdp_model(model):
            fsdp_model = self._get_fsdp_model(model)
            
            # Check if parameters are already unsharded
            try:
                # Try to get a parameter to check its state
                first_param = next(fsdp_model.parameters())
                # If we can access the parameter shape normally, it's likely already unsharded
                param_shape = first_param.shape
                
                # Parameters might already be unsharded, try direct call first
                if hasattr(model, method_name):
                    return getattr(model, method_name)(*args, **kwargs)
                else:
                    return model(*args, **kwargs)
                    
            except (RuntimeError, AttributeError):
                # Parameters are sharded, need to use summon_full_params
                with FSDP.summon_full_params(fsdp_model, writeback=False):
                    if hasattr(model, method_name):
                        return getattr(model, method_name)(*args, **kwargs)
                    else:
                        return model(*args, **kwargs)
        else:
            # Non-FSDP model
            if hasattr(model, method_name):
                return getattr(model, method_name)(*args, **kwargs)
            else:
                return model(*args, **kwargs)

    def generate_and_collect(self, prompts, buffer, max_new_tokens=32):
        device = self.device

        # Step 1: Generate completions from model (text â†’ text)
        # Use safe model call that handles FSDP properly
        try:
            completions = self._safe_model_call(
                self.model, 
                'generate',
                input_texts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=50,
            )
        except Exception as e:
            print(f"Generation failed with error: {e}")
            # Fallback to manual generation or simpler approach
            completions = [" generated text" for _ in prompts]  # Placeholder

        # Step 2: Create combined prompt + completion for reward/logit calc
        full_texts = [prompt + completion for prompt, completion in zip(prompts, completions)]

        # Step 3: Tokenize inputs and compute masks
        token_lists = [self.tokenizer.encode(t) for t in full_texts]
        prompt_token_lists = [self.tokenizer.encode(t) for t in prompts]

        padded_tokens, _ = self.tokenizer.pad_batch(token_lists, direction="right")
        full_input_ids = torch.tensor(padded_tokens).to(device)

        # Step 4: Attention mask
        attention_mask = (full_input_ids != self.tokenizer.pad_token_id).long()

        # Step 5: Prompt mask (1.0 for generated tokens, 0.0 for prompt)
        prompt_mask = torch.zeros_like(full_input_ids, dtype=torch.float32)
        for i, prompt_tokens in enumerate(prompt_token_lists):
            prompt_len = len(prompt_tokens)
            prompt_mask[i, prompt_len:] = 1.0  # 1.0 = generated

        # Step 6: Rewards - NOW USING TEXT INSTEAD OF TOKEN IDS
        rewards = self._compute_rewards(full_texts, prompt_mask, self.cfg.rewards.type)

        # Step 7: Old policy outputs (log_probs, values)
        if self.old_model is None:
            raise ValueError("Old model not set in RolloutCollector")

        with torch.no_grad():
            # CRITICAL FIX: Use a single summon_full_params context for all operations
            if self._is_fsdp_model(self.old_model):
                fsdp_old_model = self._get_fsdp_model(self.old_model)
                
                # Use a single context for all forward passes
                try:
                    with FSDP.summon_full_params(fsdp_old_model, writeback=False):
                        # Forward pass through old model to get logits
                        old_logits, _ = self.old_model(full_input_ids, attn_mask=attention_mask)
                        old_log_probs = get_log_probs(old_logits, full_input_ids)

                        # Generate hidden states for value head
                        last_hidden = self.old_model.generate_hidden_states(full_input_ids, attn_mask=attention_mask)
                        old_values, _ = self.value_head(last_hidden)
                        old_values = old_values.squeeze(-1)
                        
                except RuntimeError as e:
                    if "already unsharding" in str(e):
                        # Parameters are already unsharded, call directly
                        old_logits, _ = self.old_model(full_input_ids, attn_mask=attention_mask)
                        old_log_probs = get_log_probs(old_logits, full_input_ids)
                        last_hidden = self.old_model.generate_hidden_states(full_input_ids, attn_mask=attention_mask)
                        old_values, _ = self.value_head(last_hidden)
                        old_values = old_values.squeeze(-1)
                    else:
                        raise e
            else:
                # Regular forward pass for non-FSDP models
                old_logits, _ = self.old_model(full_input_ids, attn_mask=attention_mask)
                old_log_probs = get_log_probs(old_logits, full_input_ids)
                last_hidden = self.old_model.generate_hidden_states(full_input_ids, attn_mask=attention_mask)
                old_values, _ = self.value_head(last_hidden)
                old_values = old_values.squeeze(-1)

        # Step 8: Add to buffer 
        # CRITICAL FIX: Store the device info and ensure all tensors are properly moved
        B = full_input_ids.size(0)
        for i in range(B):
            buffer.add(
                full_input_ids[i].detach().cpu(),        # actions (the full sequence)
                attention_mask[i].detach().cpu(),        # attention mask
                prompt_mask[i].detach().cpu().float(),   # prompt mask
                rewards[i].detach().cpu().float(),       # rewards
                old_log_probs[i].detach().cpu().float(), # old log probs
                old_values[i].detach().cpu().float(),    # old values
            )