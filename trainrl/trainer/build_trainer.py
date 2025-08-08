import torch

from trainer.base_trainer import PPOTrainer
from models.model_head import ValueHead
from trainer.reward_model import HFRewardModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

def build_trainer(cfg, model):

    # model = AutoModelForCausalLM.from_pretrained(
    #     cfg["model_name"], 
    #     torch_dtype=torch.float32
    # ).cuda()
    
    value_head = ValueHead(cfg=cfg).cuda()

    # Use lower learning rate for stability
    optimizer = AdamW(
        list(model.parameters()) + list(value_head.parameters()), 
        lr=1e-5,  # Much lower learning rate
        weight_decay=0.01
    )
    
    reward_model = HFRewardModel().cuda()

    trainer = PPOTrainer(
        model=model,
        value_head=value_head,
        optimizer=optimizer,
        reward_model=reward_model,
        clip_eps=0.2,
        c1=0.5,
        c2=0.01,
        ppo_epochs=2,  # Reduce epochs for stability
        batch_size=2,  # Smaller batch size
        max_grad_norm=0.5  # More aggressive gradient clipping
    )
    
    return trainer