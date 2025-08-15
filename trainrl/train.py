import os
import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from datasets import load_dataset
from random import choices

from trainer.build_trainer import build_trainer
from trainer.buffer import RolloutBuffer
from trainer.rollout import RolloutCollector

from models.build_model import build_model

from trainer.utils import setup_signal_handlers, init_distributed_setup, cleanup_distributed
from trainer.distwrapper import build_distwrapper

def run_training(rank, world_size, cfg):
    """Function for training."""
    setup_signal_handlers(rank)

    try:
        init_distributed_setup(rank=rank, world_size=world_size)
        device = f"cuda:{rank}"
        
        model, _ = build_model(cfg=cfg)
        model = build_distwrapper(cfg=cfg, device=device, model=model, rank=rank)
        model.train()

        old_model, _ = build_model(cfg=cfg)
        old_model = build_distwrapper(cfg=cfg, device=device, model=old_model, rank=rank)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

        trainer = build_trainer(cfg=cfg, model=model, old_model=old_model)

        collector = RolloutCollector(
            model=trainer.model,
            value_head=trainer.value_head,
            reward_model=trainer.reward_model,
            device=device
        )
        collector.set_old_model(trainer.old_model)

        # 4) Prompts dataset (rank 0 loads and broadcasts, or each rank loads the same small split)
        try:
            dataset = load_dataset("OpenAssistant/oasst1", split="train")
            prompt_texts = [
                ex["text"] for ex in dataset
                if ex.get("text") and len(ex["text"]) < 200
            ][:1000]
        except Exception:
            if rank == 0:
                print("LOADING FALLBACK")
            prompt_texts = [
                "What is the capital of France?",
                "Explain photosynthesis in simple terms.",
                "Write a short story about a robot.",
                "How do you make chocolate chip cookies?",
                "What causes the seasons to change?"
            ] * 200

        if rank == 0:
            print(f"Loaded {len(prompt_texts)} prompts")

        # 5) RL loop (each rank rolls out independently; PPO trains on local shard)
        buffer = RolloutBuffer()
        update_every = 8
        num_steps = 100
        batch_size = 2
        max_new_tokens = 32

        for step in range(num_steps):
            prompts_batch = choices(prompt_texts, k=batch_size)

            collector.generate_and_collect(
                prompts=prompts_batch,
                buffer=buffer,
                max_new_tokens=max_new_tokens
            )

            if rank == 0:
                print(f"[Rank {rank} | Step {step}] Collected {len(buffer.actions)} batches")

            if (step + 1) % update_every == 0:
                if rank == 0:
                    print(f"\n[Rank {rank} | PPO Update @ step {step}] Using {len(buffer.actions)} rollouts")
                target_device = f"cuda:{rank}"
                buffer_data = buffer.get(target_device=target_device)
                
                if buffer_data is not None:
                    # Unpack the simplified buffer data
                    actions, attn_masks, prompt_masks, rewards, old_log_probs, old_values = buffer_data
                    
                    metrics = trainer.train(
                        actions=actions,
                        attention_mask=attn_masks, 
                        prompt_mask=prompt_masks,
                        rewards=rewards,
                        old_log_probs=old_log_probs,
                        old_values=old_values
                    )
                else:
                    metrics = {"loss": float('inf'), "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "mean_return": 0.0}

                if rank == 0:
                    print(f"[Train] Loss: {metrics['loss']:.2f}, "
                          f"Policy: {metrics['policy_loss']:.2f}, "
                          f"Value: {metrics['value_loss']:.2f}, "
                          f"Entropy: {metrics['entropy']:.4f}, "
                          f"Return: {metrics['mean_return']:.2f}")

                trainer.update_old_policy()
                collector.set_old_model(trainer.old_model)

                if rank == 0:
                    print("Updated old policy")

                buffer.clear()

                if metrics["loss"] > 1000:
                    if rank == 0:
                        print("Loss too high, stopping training")
                    break

    finally:
        cleanup_distributed()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    world_size = torch.cuda.device_count()

    # (optional) if you want to include hydra output dir like your other script
    if "general" in cfg and "paths" in cfg["general"] and "checkpoint_directory" in cfg["general"]["paths"]:
        hydra_working_dir = HydraConfig.get().runtime.output_dir
        cfg["general"]["paths"]["checkpoint_directory"] = os.path.join(
            hydra_working_dir, cfg["general"]["paths"]["checkpoint_directory"]
        )

    if world_size > 1:
        print(f"Launching RL training with {cfg['trainer'].get('parallelism')} on {world_size} GPUs.")
        mp.spawn(run_training, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        print("Launching RL training with 1 GPU.")
        run_training(rank=0, world_size=1, cfg=cfg)


if __name__ == "__main__":
    main()