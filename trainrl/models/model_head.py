"""
Code that will load the final linear layer, also known as the head of the model,
and perform the forward pass.
"""

import torch
import pickle
from models.utils import build_huggingface_model
from models.components.normalization import build_normalization_layer

class ModelHeadInterface(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        raise NotImplementedError
    
    def inference(self, x):
        return self.forward(x)[0][:,-1,:]
    
class HuggingFaceLMHead(ModelHeadInterface):
    def __init__(self, cfg):
        super().__init__(cfg)
        model = build_huggingface_model(cfg)
        self.lm_head = model.get_output_embeddings()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output
            None: Auxiliary output
        """
        return self.lm_head(x), None
    
class GeneralLMHead(ModelHeadInterface):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.layer_norm = build_normalization_layer(
            normalization_name=cfg["lm_head_normalization_name"],
            dim=cfg["hidden_size"],
            bias=cfg["lm_head_normalization_bias"]
        )
        self.lm_head = torch.nn.Linear(
            in_features=cfg["hidden_size"], 
            out_features=cfg["vocab_size"],
            bias=cfg["lm_head_normalization_bias"]
        )
        self.dropout = torch.nn.Dropout(
            p=cfg.get("lm_head_dropout",0.0)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.lm_head(x)
        return x, None
    

class ValueHead(ModelHeadInterface):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.value = torch.nn.Linear(cfg["hidden_size"], 1)
        # Proper initialization for value head
        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)
        torch.nn.init.constant_(self.value.bias, 0.0)
    
    def forward(self, x):
        return self.value(x), None  # output shape: (B, T, 1)