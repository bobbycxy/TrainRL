import torch

from trainer.base_trainer import PPOTrainer, ValueHead
from trainer.reward_model import HFRewardModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

def build_trainer():
    base = "Qwen/Qwen3-1.7B"  # Alternative model, or use your original "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    
    # Fix padding issue mentioned in the error
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # For generation

    model = AutoModelForCausalLM.from_pretrained(
        base, 
        torch_dtype=torch.float32
    ).cuda()
    
    value_head = ValueHead(model.config.hidden_size).cuda()

    # Use lower learning rate for stability
    optimizer = AdamW(
        list(model.parameters()) + list(value_head.parameters()), 
        lr=1e-5,  # Much lower learning rate
        weight_decay=0.01
    )
    
    reward_model = HFRewardModel().cuda()

    trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
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
    
    return trainer, tokenizer