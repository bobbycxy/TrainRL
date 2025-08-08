
import torch

class LayerNorm(torch.nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    # taken from nanoGPT
    def __init__(self, dim, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        """Apply Layer Norm"""
        return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    
NORMALIZATION_TYPE = {
    "layer_norm": lambda dim, bias: LayerNorm(dim, bias),
    "none": lambda dim, bias: torch.nn.Identity()
}

def build_normalization_layer(normalization_name, dim, bias):
    """
    Build the normalization layer
    """
    assert normalization_name in NORMALIZATION_TYPE, f"Normalization type {normalization_name} not supported"
    return NORMALIZATION_TYPE[normalization_name](dim, bias)