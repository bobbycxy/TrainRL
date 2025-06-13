# scripts/train_math_rl.py
"""
REINFORCE training script with FSDP support for large language models.
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainrl.core import RewardFunction
from trainrl.trainer import REINFORCETrainer, TrainingConfig
from trainrl.rewards import MathReward, LengthReward
from trainrl.data import load_math_dataset, create_evaluation_prompts

import torch
import torch.distributed as dist
import wandb

# Set up logging
def setup_logging(rank):
    """Setup logging for distributed training"""
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_rank_{rank}.log'),
            logging.StreamHandler()
        ] if rank == 0 else [logging.FileHandler(f'training_rank_{rank}.log')]
    )
    return logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor and log training progress with sample generations."""
    
    def __init__(self, output_dir: str, sample_prompts: list, rank: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sample_prompts = sample_prompts
        self.generation_log = []
        self.metrics_log = []
        self.rank = rank
        self.should_log = (rank == 0)  # Only rank 0 logs
    
    def log_generation(self, step: int, prompt: str, response: str, reward: float, trainer):
        """Log a sample generation."""
        if not self.should_log:
            return
            
        entry = {
            "step": step,
            "prompt": prompt,
            "response": response,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        }
        self.generation_log.append(entry)
        
        # Print to console for real-time monitoring
        print(f"\n{'='*60}")
        print(f"STEP {step} - SAMPLE GENERATION")
        print(f"{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"RESPONSE: {response}")
        print(f"REWARD: {reward:.4f}")
        print(f"BASELINE: {trainer.baseline:.4f}")
        print(f"{'='*60}\n")
    
    def log_metrics(self, step: int, metrics: dict):
        """Log training metrics."""
        if not self.should_log:
            return
            
        metrics_entry = {"step": step, **metrics, "timestamp": datetime.now().isoformat()}
        self.metrics_log.append(metrics_entry)
    
    def save_logs(self):
        """Save all logs to files."""
        if not self.should_log:
            return
            
        # Save generation log
        with open(self.output_dir / "generations.jsonl", "w") as f:
            for entry in self.generation_log:
                f.write(json.dumps(entry) + "\n")
        
        # Save metrics log
        with open(self.output_dir / "metrics.jsonl", "w") as f:
            for entry in self.metrics_log:
                f.write(json.dumps(entry) + "\n")
        
        # Save summary
        if self.metrics_log:
            summary = {
                "total_steps": len(self.metrics_log),
                "final_avg_reward": self.metrics_log[-1].get("avg_reward", 0),
                "final_baseline": self.metrics_log[-1].get("baseline", 0),
                "training_completed": datetime.now().isoformat()
            }
            with open(self.output_dir / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)


def init_distributed():
    """Initialize distributed training with FSDP support"""
    if "WORLD_SIZE" in os.environ:
        # Initialize the process group for FSDP
        dist.init_process_group(backend="nccl")
        
        # Get distributed training info
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set device for FSDP
        torch.cuda.set_device(local_rank)
        
        return world_size, rank, local_rank, True
    else:
        return 1, 0, 0, False


def load_gsm8k_data(num_samples: int = 100):
    """
    Load GSM8K dataset for math training.
    GSM8K is a dataset of grade school math word problems.
    """
    try:
        from datasets import load_dataset
        
        logger.info("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main")
        
        logger.info(f"Dataset structure: {dataset}")
        logger.info(f"Train dataset keys: {list(dataset['train'].features.keys())}")
        
        # Debug: print first example
        first_example = dataset["train"][0]
        logger.info(f"First example type: {type(first_example)}")
        logger.info(f"First example: {first_example}")
        
        # Extract training examples
        train_data = []
        train_dataset = dataset["train"]
        
        for i in range(min(num_samples, len(train_dataset))):
            example = train_dataset[i]
            
            # Handle both dict and direct access
            if isinstance(example, dict):
                question = example["question"]
                answer_text = example["answer"]
            else:
                # If it's not a dict, try direct access
                question = train_dataset[i]["question"] 
                answer_text = train_dataset[i]["answer"]
            
            # Extract the final numerical answer
            # GSM8K answers are in format "#### 123"
            answer = answer_text.split("####")[-1].strip()
            
            train_data.append({
                "prompt": f"Solve this math problem: {question}",
                "answer": answer
            })
        
        # Extract test examples  
        test_data = []
        test_dataset = dataset["test"]
        
        for i in range(min(20, len(test_dataset))):  # Small test set
            example = test_dataset[i]
            
            if isinstance(example, dict):
                question = example["question"]
                answer_text = example["answer"]
            else:
                question = test_dataset[i]["question"]
                answer_text = test_dataset[i]["answer"]
            
            answer = answer_text.split("####")[-1].strip()
            test_data.append({
                "prompt": f"Solve this math problem: {question}",
                "answer": answer
            })
        
        logger.info(f"Loaded {len(train_data)} training and {len(test_data)} test examples")
        return train_data, test_data
        
    except Exception as e:
        logger.warning(f"Failed to load GSM8K dataset: {e}")
        logger.warning("Using synthetic data instead")
        return load_synthetic_math_data(num_samples)


def load_synthetic_math_data(num_samples: int = 100):
    """
    Create synthetic math problems as fallback.
    """
    import random
    
    logger.info("Generating synthetic math data...")
    
    train_data = []
    test_data = []
    
    # Simple arithmetic problems
    operations = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b), 
        ("*", lambda a, b: a * b),
        ("//", lambda a, b: a // b if b != 0 else a)
    ]
    
    for i in range(num_samples):
        a = random.randint(1, 50)
        b = random.randint(1, 20)
        op_symbol, op_func = random.choice(operations)
        
        result = op_func(a, b)
        
        problem = {
            "prompt": f"Calculate: {a} {op_symbol} {b}",
            "answer": str(result)
        }
        
        if i < num_samples * 0.8:  # 80% train, 20% test
            train_data.append(problem)
        else:
            test_data.append(problem)
    
    logger.info(f"Generated {len(train_data)} training and {len(test_data)} test examples")
    return train_data, test_data


def create_reward_function(train_data):
    """Create the reward function that can handle multiple correct answers."""
    
    # Create a mapping from prompts to correct answers
    answer_map = {item["prompt"]: item["answer"] for item in train_data}
    
    class DynamicMathReward:
        def __init__(self, answer_map):
            self.answer_map = answer_map
            self.length_reward = LengthReward(target_length=150, tolerance=50)
        
        def __call__(self, prompt, response):
            # Get the correct answer for this prompt
            correct_answer = self.answer_map.get(prompt, '42')
            
            # Create math reward for this specific answer
            math_reward = MathReward(correct_answer=correct_answer)
            
            # Combine rewards
            math_score = math_reward(prompt, response)
            length_score = self.length_reward(prompt, response)
            
            final_score = 0.8 * math_score + 0.2 * length_score
            return final_score
    
    return DynamicMathReward(answer_map)


def monitor_training_progress(trainer, monitor_prompts, step, monitor):
    """Generate sample responses and log them for monitoring."""
    for prompt in monitor_prompts[:2]:  # Monitor first 2 prompts
        response, _, _ = trainer.generate_response(prompt, do_sample=True)
        reward = trainer.reward_function(prompt, response)
        monitor.log_generation(step, prompt, response, reward, trainer)


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train LLM with REINFORCE on math problems using FSDP")
    parser.add_argument("--model", default="gpt2", help="Model to train")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--output_dir", default="./training_output", help="Output directory")
    parser.add_argument("--monitor_every", type=int, default=50, help="Monitor generations every N steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save model every N steps") 
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps")
    
    # FSDP specific arguments
    parser.add_argument("--sharding_strategy", default="FULL_SHARD", 
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                        help="FSDP sharding strategy")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Enable mixed precision training")
    parser.add_argument("--cpu_offload", action="store_true", 
                        help="Offload parameters to CPU")
    parser.add_argument("--activation_checkpointing", action="store_true",
                        help="Enable activation checkpointing")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()

    # Initialize distributed training
    world_size, rank, local_rank, is_distributed = init_distributed()
    
    # Setup logging
    global logger
    logger = setup_logging(rank)
    
    if rank == 0:
        logger.info(f"Distributed training: {is_distributed}")
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        logger.info(f"FSDP Configuration:")
        logger.info(f"  - Sharding strategy: {args.sharding_strategy}")
        logger.info(f"  - Mixed precision: {args.mixed_precision}")
        logger.info(f"  - CPU offload: {args.cpu_offload}")
        logger.info(f"  - Activation checkpointing: {args.activation_checkpointing}")

    # Initialize Weights & Biases (only on rank 0)
    if rank == 0 and not args.no_wandb:
        wandb.init(
            project="trainrl-fsdp", 
            entity="i2r-ali", 
            config=args, 
            name=f"fsdp_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"fsdp_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting FSDP training run in {output_dir}")
    
    # Synchronize all processes
    if is_distributed:
        dist.barrier()
    
    # Load data
    train_data, test_data = load_gsm8k_data(args.num_samples)
    
    # Extract prompts for training
    train_prompts = [item["prompt"] for item in train_data]
    test_prompts = [item["prompt"] for item in test_data]
    
    # Create sample prompts for monitoring (first 3 test examples)
    monitor_prompts = test_prompts[:3]
    
    # Create reward function with access to all training data answers
    reward_function = create_reward_function(train_data)
    
    # Training configuration with FSDP settings
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=256,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_clip=1.0,
        baseline_decay=0.95,
        # FSDP specific settings
        sharding_strategy=args.sharding_strategy,
        mixed_precision=args.mixed_precision,
        cpu_offload=args.cpu_offload,
        activation_checkpointing=args.activation_checkpointing
    )
    
    # Initialize trainer
    if rank == 0:
        logger.info(f"Initializing FSDP trainer with model: {args.model}")
    trainer = REINFORCETrainer(
        model_name=args.model,
        reward_function=reward_function,
        config=config
    )
    
    # Print model info (only on rank 0)
    if rank == 0:
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Print FSDP info
        if is_distributed:
            logger.info(f"FSDP model type: {type(trainer.model)}")
            if hasattr(trainer.model, 'sharding_strategy'):
                logger.info(f"FSDP sharding strategy: {trainer.model.sharding_strategy}")
    
    # Initialize monitor
    monitor = TrainingMonitor(output_dir, monitor_prompts, rank)
    
    # Save configuration (only on rank 0)
    if rank == 0:
        config_dict = {
            "model": args.model,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "world_size": world_size,
            "sharding_strategy": args.sharding_strategy,
            "mixed_precision": args.mixed_precision,
            "cpu_offload": args.cpu_offload,
            "activation_checkpointing": args.activation_checkpointing,
            "training_start": datetime.now().isoformat()
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    
    if rank == 0:
        logger.info("Starting FSDP training with trainer.train() method...")
    
    # Custom callback function to monitor training progress
    class TrainingCallback:
        def __init__(self, monitor, monitor_prompts, monitor_every, rank, no_wandb):
            self.monitor = monitor
            self.monitor_prompts = monitor_prompts
            self.monitor_every = monitor_every
            self.step = 0
            self.rank = rank
            self.no_wandb = no_wandb
        
        def on_step_end(self, trainer, metrics):
            self.step += 1
            
            # Log metrics
            self.monitor.log_metrics(self.step, metrics)
            
            # Log to wandb (only rank 0)
            if self.rank == 0 and not self.no_wandb:
                wandb.log({
                    "step": self.step,
                    "loss": metrics['loss'],
                    "avg_reward": metrics['avg_reward'],
                    "baseline": metrics['baseline'],
                    "avg_advantage": metrics.get('avg_advantage', 0)
                })
            
            # Monitor generations (only rank 0)
            if self.rank == 0 and self.step % self.monitor_every == 0:
                monitor_training_progress(trainer, self.monitor_prompts, self.step, self.monitor)
    
    # Create callback
    callback = TrainingCallback(monitor, monitor_prompts, args.monitor_every, rank, args.no_wandb)
    
    # Monkey patch the trainer to add callbacks (simple approach)
    original_train_step = trainer.train_step
    def train_step_with_callback(prompts):
        metrics = original_train_step(prompts)
        callback.on_step_end(trainer, metrics)
        return metrics
    trainer.train_step = train_step_with_callback
    
    # Run training using trainer.train() method
    try:
        if rank == 0:
            logger.info("Starting FSDP training...")
        
        history = trainer.train(
            train_prompts=train_prompts,
            eval_prompts=test_prompts[:10]  # Use subset for evaluation
        )
        
        if rank == 0:
            logger.info("FSDP training completed successfully!")
            logger.info(f"Training history keys: {history.keys()}")
        
    except Exception as e:
        logger.error(f"FSDP training failed: {e}")
        cleanup_distributed()
        raise
    
    # Final evaluation
    if rank == 0:
        logger.info("Running final evaluation...")
    try:
        eval_metrics = trainer.evaluate(test_prompts[:10])  # Evaluate on subset
        if rank == 0:
            logger.info(f"Final evaluation: {eval_metrics}")
            
            # Log final evaluation to wandb
            if not args.no_wandb:
                wandb.log({
                    "final_eval_reward": eval_metrics.get('eval_avg_reward', 0),
                    "training_complete": True
                })
        
    except Exception as e:
        if rank == 0:
            logger.warning(f"Final evaluation failed: {e}")
        eval_metrics = {"eval_avg_reward": "N/A"}
    
    # Save final model (only on rank 0)
    if rank == 0:
        model_path = output_dir / "final_model"
        try:
            logger.info("Saving FSDP model...")
            trainer.save_model(str(model_path))
            logger.info(f"FSDP model saved to {model_path}")
        except Exception as e:
            logger.warning(f"Failed to save FSDP model: {e}")
            # Manual save as fallback
            try:
                import torch
                # Save what we can
                model_path.mkdir(exist_ok=True)
                torch.save({
                    'baseline': trainer.baseline,
                    'config': config_dict,
                    'optimizer_state_dict': trainer.optimizer.state_dict() if hasattr(trainer.optimizer, 'state_dict') else None,
                }, model_path / "training_state.pt")
                
                # Save tokenizer
                trainer.tokenizer.save_pretrained(str(model_path))
                logger.info(f"Partial model state saved to {model_path}")
            except Exception as e2:
                logger.error(f"Manual save also failed: {e2}")
    
    # Synchronize before saving logs
    if is_distributed:
        dist.barrier()
    
    # Save all logs (only on rank 0)
    if rank == 0:
        monitor.save_logs()
        logger.info(f"Training logs saved to {output_dir}")
        
        # Save training history
        with open(output_dir / "training_history.json", "w") as f:
            # Convert any tensors to floats for JSON serialization
            serializable_history = {}
            for key, values in history.items():
                serializable_history[key] = [float(v) if hasattr(v, 'item') else v for v in values]
            json.dump(serializable_history, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("FSDP TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Final average reward: {eval_metrics.get('eval_avg_reward', 'N/A')}")
        print(f"Total training steps: {callback.step}")
        print(f"Training epochs: {args.num_epochs}")
        print(f"World size: {world_size}")
        print(f"Sharding strategy: {args.sharding_strategy}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"CPU offload: {args.cpu_offload}")
        print(f"Check {output_dir}/generations.jsonl for response samples")
        print(f"Check {output_dir}/metrics.jsonl for training metrics")
        print(f"Check {output_dir}/training_history.json for full training history")
        print(f"{'='*60}")
        
        # Finish wandb run
        if not args.no_wandb:
            wandb.finish()
    
    # Clean up distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()