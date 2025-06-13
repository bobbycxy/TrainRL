import os
import re
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from functools import partial
import numpy as np
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm
from enum import Enum

from trainrl import RLConfig, RLAlgorithm, PPOTrainer, REINFORCETrainer, BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSM8KDataset(Dataset):
    """Dataset wrapper for GSM8K mathematical reasoning problems"""
    
    def __init__(self, tokenizer, max_length=512, split="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load GSM8K dataset
        dataset = load_dataset("gsm8k", "main", split=split)
        self.data = []
        
        for item in dataset:
            question = item["question"]
            answer = item["answer"]
            
            # Extract the numerical answer
            numerical_answer = self.extract_numerical_answer(answer)
            
            self.data.append({
                "question": question,
                "answer": answer,
                "numerical_answer": numerical_answer,
                "prompt": f"Question: {question}\nAnswer:"
            })
    
    def extract_numerical_answer(self, answer_text):
        """Extract the final numerical answer from GSM8K answer text"""
        # GSM8K answers typically end with "#### NUMBER"
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
        if match:
            return float(match.group(1))
        
        # Fallback: look for numbers at the end
        numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
        if numbers:
            return float(numbers[-1])
        
        return 0.0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize the prompt
        encoding = self.tokenizer(
            item["prompt"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "question": item["question"],
            "correct_answer": item["numerical_answer"],
            "full_answer": item["answer"]
        }


def setup_distributed():
    """Initialize distributed training environment"""
    if not dist.is_initialized():
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"])
        )
    
    torch.cuda.set_device(dist.get_rank())


def create_trainer(model, tokenizer, config: RLConfig):
    """Factory function to create the appropriate trainer based on config"""
    if config.rl_algorithm == RLAlgorithm.REINFORCE:
        if dist.get_rank() == 0:
            logger.info("Using REINFORCE trainer")
        return REINFORCETrainer(model, tokenizer, config)
    elif config.rl_algorithm == RLAlgorithm.PPO:
        if dist.get_rank() == 0:
            logger.info("Using PPO trainer")
        return PPOTrainer(model, tokenizer, config)
    else:
        raise ValueError(f"Unknown RL algorithm: {config.rl_algorithm}")


def main():
    """Main training function"""
    setup_distributed()
    
    config = RLConfig()
    
    # Toggle between algorithms here:
    # config.rl_algorithm = RLAlgorithm.REINFORCE  # Change to RLAlgorithm.PPO for PPO
    config.rl_algorithm = RLAlgorithm.PPO  # Default to PPO for this example
    
    if dist.get_rank() == 0:
        logger.info(f"Starting RL training with model: {config.model_name}")
        logger.info(f"Using algorithm: {config.rl_algorithm.value}")
    
    # Load tokenizer and model
    base_model = BaseModel(config.model_name, fsdp=True)
    base_model.freeze_all_layers()
    base_model.unfreeze_last_n_layers(n=2)

    model = base_model.model
    tokenizer = base_model.tokenizer
    
    if dist.get_rank() == 0:
        logger.info("Model wrapped with FSDP successfully")
    
    # Create dataset and dataloader
    dataset = GSM8KDataset(tokenizer, config.max_length, split="train[:1000]")  # Small subset for demo
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=dist.get_world_size(), 
        rank=dist.get_rank()
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Create trainer based on config
    trainer = create_trainer(model, tokenizer, config)
    
    if dist.get_rank() == 0:
        logger.info(f"Starting training for {config.num_epochs} epochs")
    
    # Training loop
    for epoch in range(config.num_epochs):
        sampler.set_epoch(epoch)
        trainer.model.train()
        
        total_loss = 0
        total_reward = 0
        num_batches = 0
        
        if dist.get_rank() == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        else:
            pbar = dataloader
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            metrics = trainer.train_step(batch)
            
            total_loss += metrics["loss"]
            total_reward += metrics["reward"]
            num_batches += 1
            
            if dist.get_rank() == 0:
                # Create progress bar description based on trainer type
                postfix_dict = {
                    "Loss": f"{metrics['loss']:.4f}",
                    "Reward": f"{metrics['reward']:.2f}"
                }
                
                # Add algorithm-specific metrics
                if isinstance(trainer, REINFORCETrainer):
                    postfix_dict["Baseline"] = f"{metrics['baseline']:.2f}"
                    postfix_dict["Advantage"] = f"{metrics['advantage']:.2f}"
                elif isinstance(trainer, PPOTrainer):
                    postfix_dict["PolicyLoss"] = f"{metrics['policy_loss']:.4f}"
                    if "value_loss" in metrics:
                        postfix_dict["ValueLoss"] = f"{metrics['value_loss']:.4f}"
                
                pbar.set_postfix(postfix_dict)
                
                # Log some examples
                if num_batches % 10 == 0:
                    logger.info(f"\nSample generations ({config.rl_algorithm.value}):")
                    for q, gen, r in metrics["generated_samples"]:
                        logger.info(f"Q: {q[:100]}...")
                        logger.info(f"A: {gen[:200]}...")
                        logger.info(f"Reward: {r:.2f}\n")
        
        # Log epoch results
        if dist.get_rank() == 0:
            avg_loss = total_loss / num_batches
            avg_reward = total_reward / num_batches
            logger.info(f"Epoch {epoch+1} completed ({config.rl_algorithm.value}) - Loss: {avg_loss:.4f}, Reward: {avg_reward:.2f}")
    
    if dist.get_rank() == 0:
        logger.info("Training completed!")
        
        # Save model (in practice, you'd want to save checkpoints during training)
        logger.info("Saving final model...")
        # Note: Saving FSDP models requires special handling
        # You might want to use FSDP's state_dict utilities
    
    dist.barrier()
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()