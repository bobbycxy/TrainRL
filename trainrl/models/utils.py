import os
import torch
from transformers import AutoModelForCausalLM

def build_huggingface_model(model_cfg, freeze = True):
    """
    Build the huggingface model and optionally freeze the parameters. 
    """

    model_str = model_cfg.model_name

    flash_attention = model_cfg.get("flash_attention", False)
    if flash_attention:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        attn_implementation=attn_implementation,
        token=os.getenv("HF_ACCESS_TOKEN"),
        low_cpu_mem_usage=True,
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "float32"))
    )

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model