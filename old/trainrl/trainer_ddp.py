import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .core import RewardFunction

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_length: int = 512
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    gradient_clip: float = 1.0
    baseline_decay: float = 0.99
    temperature: float = 1.0

class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

class REINFORCETrainer:
    def __init__(self, model_name: str, reward_function: RewardFunction, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.reward_function = reward_function

        # Initialize distributed training if available
        self.is_distributed = dist.is_available() and dist.is_initialized()
        
        if self.is_distributed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        
        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.baseline = 0.0

    def _get_unwrapped_model(self):
        """Get the unwrapped model (useful for DDP)"""
        return self.model.module if self.is_distributed else self.model

    def generate_response(self, prompt: str, do_sample: bool = True) -> Tuple[str, torch.Tensor, torch.Tensor]:
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_tokens.shape[1]

        with torch.no_grad():
            outputs = self._get_unwrapped_model().generate(
                prompt_tokens,
                max_length=min(prompt_length + 200, self.config.max_length),
                do_sample=do_sample,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True if do_sample else False
            )
            generated_tokens = outputs.sequences[0][prompt_length:]

        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if do_sample:
            full_tokens = torch.cat([prompt_tokens, generated_tokens.unsqueeze(0)], dim=1)
            log_probs = self._compute_log_probs(full_tokens, prompt_length)
        else:
            log_probs = torch.zeros(len(generated_tokens), device=self.device)

        return response_text, log_probs, generated_tokens

    def _compute_log_probs(self, tokens: torch.Tensor, prompt_length: int) -> torch.Tensor:
        outputs = self.model(tokens)
        logits = outputs.logits
        generated_logits = logits[0, prompt_length - 1:-1]
        generated_tokens = tokens[0, prompt_length:]
        log_probs = torch.log_softmax(generated_logits, dim=-1)
        return log_probs.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)

    def _aggregate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate metrics across all processes in distributed training"""
        if not self.is_distributed:
            return metrics
        
        aggregated_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            aggregated_metrics[key] = (tensor / self.world_size).item()
        
        return aggregated_metrics

    def _should_log(self) -> bool:
        """Check if current process should log (only rank 0 in distributed)"""
        return not self.is_distributed or self.rank == 0

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        self.model.train()
        batch_rewards = []
        batch_log_probs = []

        # Generate responses and compute rewards
        for prompt in prompts:
            response, log_probs, _ = self.generate_response(prompt, do_sample=True)
            reward = self.reward_function(prompt, response)
            batch_rewards.append(reward)
            batch_log_probs.append(log_probs)

        # Update baseline
        current_avg_reward = sum(batch_rewards) / len(batch_rewards)
        self.baseline = self.config.baseline_decay * self.baseline + (1 - self.config.baseline_decay) * current_avg_reward

        # Compute loss
        total_loss = 0
        advantages = []
        for reward, log_probs in zip(batch_rewards, batch_log_probs):
            advantage = reward - self.baseline
            advantages.append(advantage)
            if len(log_probs) > 0:  # Ensure we have generated tokens
                loss = -log_probs.sum() * advantage
                total_loss += loss

        if len(batch_rewards) > 0:
            total_loss = total_loss / len(batch_rewards)
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        # Prepare local metrics
        local_metrics = {
            "loss": total_loss.item(),
            "avg_reward": current_avg_reward,
            "baseline": self.baseline,
            "avg_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
            "batch_size": len(prompts)
        }

        # Aggregate metrics across all processes
        aggregated_metrics = self._aggregate_metrics(local_metrics)

        return aggregated_metrics

    def evaluate(self, eval_prompts: List[str]) -> Dict[str, float]:
        """Evaluate the model on evaluation prompts"""
        self.model.eval()
        eval_rewards = []
        
        with torch.no_grad():
            for prompt in eval_prompts:
                response, _, _ = self.generate_response(prompt, do_sample=False)
                reward = self.reward_function(prompt, response)
                eval_rewards.append(reward)
        
        local_eval_metrics = {
            "eval_avg_reward": sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0,
            "eval_samples": len(eval_rewards)
        }
        
        # Aggregate evaluation metrics
        aggregated_eval_metrics = self._aggregate_metrics(local_eval_metrics)
        
        return aggregated_eval_metrics

    def train(self, train_prompts: List[str], eval_prompts: Optional[List[str]] = None) -> Dict[str, List[float]]:
        history = {"loss": [], "avg_reward": [], "baseline": [], "eval_reward": [], "avg_advantage": []}

        # Create dataset and dataloader
        train_dataset = PromptDataset(train_prompts)
        
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=True
            )
            train_loader = DataLoader(
                train_dataset, 
                sampler=train_sampler, 
                batch_size=self.config.batch_size,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size,
                shuffle=True,
                pin_memory=True
            )

        total_steps = 0

        for epoch in range(self.config.num_epochs):
            if self.is_distributed:
                train_sampler.set_epoch(epoch)
            
            epoch_progress = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                disable=not self._should_log()
            )
            
            for step, batch_prompts in enumerate(epoch_progress):
                # Convert batch_prompts to list if it's a tensor
                if isinstance(batch_prompts, torch.Tensor):
                    batch_prompts = [train_prompts[i] for i in batch_prompts]
                
                metrics = self.train_step(batch_prompts)
                total_steps += 1

                # Update progress bar
                if self._should_log():
                    epoch_progress.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'reward': f"{metrics['avg_reward']:.4f}",
                        'baseline': f"{metrics['baseline']:.4f}"
                    })

                # Log metrics
                if total_steps % self.config.logging_steps == 0:
                    if self._should_log():
                        logger.info(f"Epoch {epoch+1}, Step {total_steps}: {metrics}")
                    
                    # Store history (only on rank 0 to avoid duplication)
                    if self._should_log():
                        history["loss"].append(metrics["loss"])
                        history["avg_reward"].append(metrics["avg_reward"])
                        history["baseline"].append(metrics["baseline"])
                        history["avg_advantage"].append(metrics["avg_advantage"])

                # Evaluation
                if eval_prompts and total_steps % (self.config.logging_steps * 2) == 0:
                    eval_metrics = self.evaluate(eval_prompts)
                    if self._should_log():
                        logger.info(f"Evaluation at step {total_steps}: {eval_metrics}")
                        history["eval_reward"].append(eval_metrics["eval_avg_reward"])

        # Final evaluation
        if eval_prompts:
            final_eval_metrics = self.evaluate(eval_prompts)
            if self._should_log():
                logger.info(f"Final evaluation: {final_eval_metrics}")
                history["eval_reward"].append(final_eval_metrics["eval_avg_reward"])

        return history

    def save_model(self, save_path: str):
        """Save the model (only on rank 0)"""
        if self._should_log():
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            unwrapped_model = self._get_unwrapped_model()
            unwrapped_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save additional training state
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'baseline': self.baseline,
                'config': self.config
            }, os.path.join(save_path, "training_state.pt"))
            
            logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """Load the model"""
        # Load model and tokenizer
        unwrapped_model = self._get_unwrapped_model()
        unwrapped_model.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        # Load additional training state
        state_path = os.path.join(load_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.baseline = state['baseline']
            logger.info(f"Training state loaded from {state_path}")
        
        logger.info(f"Model loaded from {load_path}")