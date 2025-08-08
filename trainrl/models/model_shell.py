"""
Create the shell of the model.

It will take in the embedding model, core model, and model head.

Of which, this class will then be able to perform forward passes over a given input, 
perform inference, generate.
"""

import torch

class ModelShell(torch.nn.Module):
    def __init__(self, embedding_model, core_model, model_head, model_cfg):
        super().__init__()
        self.embedding_model = embedding_model
        self.core_model = core_model
        self.model_head = model_head
        self.model_cfg = model_cfg

    def to(self, device):
        self.embedding_model.to(device)
        self.core_model.to(device)
        self.model_head.to(device)
        return super().to(device)

    def forward(self, token_ids, attn_mask=None):
        """
        Forward pass of the model
        """
        embeddings = self.embedding_model(token_ids)
        hidden_states = self.core_model(embeddings, attn_mask)
        logits, _ = self.model_head(hidden_states)
        return logits, _

    @torch.no_grad()
    def inference(self, model_input):
        """
        Perform inference
        """
        if isinstance(model_input, str):
            model_input = self.embedding_model.tokenize_input(model_input)
            model_input = torch.tensor(model_input).unsqueeze(0)
        embeddings = self.embedding_model(model_input)
        hidden_states = self.core_model(embeddings)
        logits = self.model_head.inference(hidden_states)
        return logits
    
    @torch.no_grad()
    def generate_hidden_states(self, model_input, attn_mask=None):
        """
        Generate embeddings
        """
        if isinstance(model_input, str):
            model_input = self.embedding_model.tokenize_input(model_input)
            model_input = torch.tensor(model_input).unsqueeze(0)
        embeddings = self.embedding_model(model_input)
        hidden_states = self.core_model(embeddings, attn_mask)
        return hidden_states
    
    @torch.no_grad()
    def generate(
        self,
        input_texts,
        max_new_tokens=128,
        temperature=1.0,
        top_k=None,
    ):
        """
        Autoregressive generation for single or batched input texts.
        
        Args:
            input_texts (str or List[str])
            max_new_tokens (int)
            temperature (float)
            top_k (int or None): top-k sampling. Set to 1 for greedy.
        
        Returns:
            List[str]: Decoded generated texts
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # Step 1: Tokenize
        token_lists = [self.embedding_model.tokenize_input(t, add_eot=False) for t in input_texts]
        token_lists = self.embedding_model.truncate(token_lists, truncate_from="right")
        input_ids, _ = self.embedding_model.pad_batch(token_lists, direction="right")
        input_ids = torch.tensor(input_ids).to(next(self.parameters()).device)  # [B, T]

        batch_size = input_ids.size(0)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            embeddings = self.embedding_model(input_ids)  # [B, T, D]
            hidden_states = self.core_model(embeddings)   # [B, T, D]
            logits = self.model_head.inference(hidden_states)  # [B, vocab]

            # Step 2: Apply temperature
            logits = logits / temperature

            # Step 3: Top-k filtering
            if top_k is not None:
                top_k_logits, _ = torch.topk(logits, k=top_k, dim=-1)
                min_top_k_logit = top_k_logits[:, -1].unsqueeze(-1)
                logits[logits < min_top_k_logit] = -float("inf")

            # Step 4: Sampling / greedy
            if top_k == 1:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)     # [B, 1]

            # Step 5: Track finished sequences
            is_eot = next_token.squeeze(-1) == self.embedding_model.eot_token_id
            is_finished |= is_eot

            # Replace future tokens with pad for finished sequences
            next_token = torch.where(
                is_finished.unsqueeze(-1),
                torch.full_like(next_token, self.embedding_model.pad_token_id),
                next_token
            )

            # Step 6: Append next token
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # [B, T+1]

            if is_finished.all():
                break

        # Step 7: Decode
        return self.embedding_model.decode(input_ids.tolist())

