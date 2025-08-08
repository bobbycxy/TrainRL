import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from random import choices

from trainer.build_trainer import build_trainer
from trainer.buffer import RolloutBuffer
from trainer.rollout import RolloutCollector

from models.build_model import build_model


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    model, _ = build_model(cfg=cfg)
    
    model = model.to(device="cuda")

    trainer = build_trainer(cfg=cfg, model=model)

    collector = RolloutCollector(
        model=trainer.model,
        value_head=trainer.value_head,
        reward_model=trainer.reward_model,
        device="cuda"
    )
    collector.set_old_model(trainer.old_model)
    
    # Load dataset
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        prompt_texts = [
            example["text"] for example in dataset
            if example.get("text") and len(example["text"]) < 200  # Shorter prompts
        ][:1000]  # Limit dataset size
    except:
        # Fallback simple prompts if dataset loading fails
        print("LOADING FALLBACK")
        prompt_texts = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a short story about a robot.",
            "How do you make chocolate chip cookies?",
            "What causes the seasons to change?"
        ] * 200

    print(f"Loaded {len(prompt_texts)} prompts")
    
    buffer = RolloutBuffer()
    update_every = 8  # Number of batches before PPO update
    num_steps = 100  # Reduced for testing
    batch_size = 2
    max_new_tokens = 32

    for step in range(num_steps):
        # Sample prompts
        prompts_batch = choices(prompt_texts, k=batch_size)

        # Generate data and accumulate in buffer (no training yet)
        collector.generate_and_collect(
            prompts=prompts_batch,
            buffer=buffer,
            max_new_tokens=max_new_tokens
        )

        print(f"[Step {step}] Collected {len(buffer.input_ids)} batches")

                # Update the policy every `update_every` batches
        if (step + 1) % update_every == 0:
            print(f"\n[Training Step {step}] Updating policy using {len(buffer.input_ids)} rollouts")

            metrics = trainer.train(*buffer.get())

            print(f"[Train] Loss: {metrics['loss']:.2f}, Policy: {metrics['policy_loss']:.2f}, "
                  f"Value: {metrics['value_loss']:.2f}, Entropy: {metrics['entropy']:.4f}, Return: {metrics['mean_return']:.2f}")

            trainer.update_old_policy()
            collector.set_old_model(trainer.old_model)
            print("Updated old policy")

            buffer.clear()

            if metrics["loss"] > 1000:
                print("Loss too high, stopping training")
                break

if __name__ == "__main__":
    main()