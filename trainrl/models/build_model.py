"""
This puts together the embedding model, core model, and the model head.

Then the model will use the model_shell to be the skeleton to put together all three components together.

The model will then be able to perform forward passes over a given input, perform inference, generate.
"""

from models.core_model import HuggingFaceTransformerCore, HuggingFaceWithMLPTransformerCore
from models.embedding_model import HuggingFaceEmbedder
from models.model_head import HuggingFaceLMHead, GeneralLMHead
from models.model_shell import ModelShell

import torch

EmbeddingDict = {
    "huggingface": lambda cfg :HuggingFaceEmbedder(cfg)
}

def build_embedding_model(cfg):
    """
    Build the embedding model
    """
    embedding_type = cfg.get("embedding_model_type", "huggingface") ## to do up
    return EmbeddingDict[embedding_type](cfg)



CoreModelDict = {
    "huggingface": lambda cfg :HuggingFaceTransformerCore(cfg),
    "huggingface_with_mlp": lambda cfg :HuggingFaceWithMLPTransformerCore(cfg)
}

def build_core_model(cfg):
    """
    Build the core model
    """
    core_model_type = cfg.get("core_model_type", "huggingface")
    return CoreModelDict[core_model_type](cfg)



ModelHeadDict = {
    "huggingface": lambda cfg :HuggingFaceLMHead(cfg),
    "general": lambda cfg :GeneralLMHead(cfg),
}

def build_model_head(cfg):
    """
    Build the model head
    """
    model_head_type = cfg.get("model_head_type", "huggingface")
    return ModelHeadDict[model_head_type](cfg)



ModelShellDict = {
    "standard": lambda embedding_model, core_model, model_head, cfg :ModelShell(embedding_model, core_model, model_head, cfg),
}

def build_model_shell(embedding_model, core_model, model_head, cfg):
    """
    Build the model shell
    """
    model_shell_type = "standard"
    return ModelShellDict[model_shell_type](embedding_model, core_model, model_head, cfg)



def initialize_model(cfg):
    """
    Initialize the model
    """
    embedding_model = build_embedding_model(cfg)
    core_model = build_core_model(cfg)
    model_head = build_model_head(cfg)
    model = build_model_shell(embedding_model, core_model, model_head, cfg)
    return model


def build_model(cfg=None, checkpoint_path=None, device="cuda"):
    """
    Build the model
    """
    if checkpoint_path:

        ## fetch the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model = initialize_model(checkpoint["cfg"])

        model.load_state_dict(checkpoint["model"])

        loaded_train_config = {
            "optimizer": checkpoint["optimizer"],
            "lr_scheduler": checkpoint["lr_scheduler"],
            "current_iter": checkpoint["current_iter"],
            "mp_config": checkpoint["mp_config"]
        }

        return model, loaded_train_config

    else:

        model = initialize_model(cfg)

        if "torch_dtype" in cfg:
            model = model.to(dtype=getattr(torch, cfg["torch_dtype"]))
        
        return model, None