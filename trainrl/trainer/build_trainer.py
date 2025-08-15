import torch

from trainer.base_trainer import PPOTrainer
from models.model_head import ValueHead
from trainer.reward_model import HFRewardModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def build_trainer(cfg, model, old_model):

    # model = AutoModelForCausalLM.from_pretrained(
    #     cfg["model_name"], 
    #     torch_dtype=torch.float32
    # ).cuda()
    
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    value_head = ValueHead(cfg=cfg).cuda()
    value_head = FSDP(value_head, device_id=rank)
    

    # Use lower learning rate for stability
    optimizer = AdamW(
        list(model.parameters()) + list(value_head.parameters()), 
        lr=1e-4,  # Much lower learning rate
        weight_decay=0.01
    )
    
    reward_model = HFRewardModel()
    reward_model.model.to(f"cuda:{rank}")
    reward_model.model.eval()

    trainer = PPOTrainer(
        model=model,
        old_model=old_model,
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