import torch
from trainer.utils import get_log_probs

class RolloutCollector:
    def __init__(self, model, value_head, reward_model=None, device="cuda"):
        self.model = model
        self.tokenizer = model.embedding_model.tokenizer
        self.value_head = value_head
        self.reward_model = reward_model
        self.device = device

        self.old_model = None  # Optional: explicitly settable, otherwise fallback to deepcopy(model)
        self._log_prob_fn = None  # Optional: set externally

    def set_old_model(self, old_model):
        self.old_model = old_model
        self.old_model.cuda()

    def _compute_rewards(self, input_ids, prompt_mask):
        with torch.no_grad():
            rewards = self.reward_model(input_ids)
            if rewards.numel() > 0:
                rewards = rewards * prompt_mask
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards

    def generate_and_collect(self, prompts, buffer, max_new_tokens=32):
        device = self.device

        # Step 1: Generate completions from model (text → text)
        completions = self.model.generate(
            input_texts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
        )  # List[str]

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

        # Step 6: Rewards
        rewards = self._compute_rewards(full_input_ids, prompt_mask)

        # Step 7: Old policy outputs (log_probs, values)
        if self.old_model is None:
            raise ValueError("Old model not set in RolloutCollector")

        with torch.no_grad():
            # Forward pass through old model to get logits
            old_logits, _ = self.old_model(full_input_ids, attn_mask=attention_mask)  # [B, T, V]
            old_log_probs = get_log_probs(old_logits, full_input_ids)              # [B, T]

            # Generate hidden states for value head
            last_hidden = self.old_model.generate_hidden_states(full_input_ids, attn_mask=attention_mask)  # [B, T, H]
            old_values, _ = self.value_head(last_hidden)   # [B, T, 1]
            old_values = old_values.squeeze(-1)            # [B, T]

        # Step 8: Add to buffer
        B = full_input_ids.size(0)
        for i in range(B):
            buffer.add(
                full_input_ids[i],
                attention_mask[i],
                prompt_mask[i],
                full_input_ids[i],
                rewards[i],
                old_log_probs[i],
                old_values[i],
            )
