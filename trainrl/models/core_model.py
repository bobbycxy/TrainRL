"""
Code that will prepare only the core transformer architecture of the given model.
"""
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from models.utils import build_huggingface_model

class CoreInterface(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, x, attn_mask=None):
        raise NotImplementedError

class HuggingFaceTransformerCore(CoreInterface):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        model = build_huggingface_model(model_cfg)

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("[Checkpointing] Enabled HF gradient checkpointing.")

        delattr(model.model, "embed_tokens")

        self.model = model.model 

    def forward(self, x, attn_mask=None):
        hidden_states = self.model(
            inputs_embeds=x,
            attention_mask=attn_mask,
            output_hidden_states=True
        ).hidden_states

        # return last_hidden_states
        if isinstance(hidden_states, tuple):
            last_hidden_states = hidden_states[-1]
            return last_hidden_states
        
class HuggingFaceWithMLPTransformerCore(CoreInterface):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        model = build_huggingface_model(model_cfg)

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("[Checkpointing] Enabled HF gradient checkpointing.")

        delattr(model.model, "embed_tokens")

        self.model = model.model  
        
        # Define an MLP with a residual connection
        self.mlp = nn.Sequential(
            nn.Linear(model_cfg["hidden_size"], 4 * model_cfg["hidden_size"]),
            nn.ReLU(),
            nn.Linear(4 * model_cfg["hidden_size"], model_cfg["hidden_size"])
        )
        
        # Layer normalization before returning output (optional but recommended)
        self.layer_norm = nn.LayerNorm(model_cfg["hidden_size"])

    def forward(self, x, attn_mask=None):
        hidden_states = self.model(
            inputs_embeds=x,
            attention_mask=attn_mask,
            output_hidden_states=True
        ).hidden_states

        # Get last hidden states
        if isinstance(hidden_states, tuple):
            last_hidden_states = hidden_states[-1]

        # Pass through MLP with residual connection
        transformed_states = self.mlp(last_hidden_states)
        last_hidden_states = last_hidden_states + transformed_states  # Residual connection

        # Apply layer normalization
        last_hidden_states = self.layer_norm(last_hidden_states)

        return last_hidden_states  