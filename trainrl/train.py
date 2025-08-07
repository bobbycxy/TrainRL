import torch
from datasets import load_dataset
from random import choices

from trainer.build_trainer import build_trainer
from trainer.base_trainer import RolloutBuffer

def generate_and_collect(prompts, trainer, tokenizer, buffer, max_new_tokens=32):
    # Same setup as before
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    input_ids = encodings.input_ids.cuda()
    attention_mask = encodings.attention_mask.cuda()
    prompt_lens = attention_mask.sum(dim=1).tolist()

    with torch.no_grad():
        generation = trainer.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        generated_ids = generation.sequences[:, input_ids.shape[1]:]

    # Combine prompt + generation
    full_input_ids = torch.cat([input_ids, generated_ids], dim=1)
    full_attention_mask = torch.cat([
        attention_mask,
        torch.ones_like(generated_ids)
    ], dim=1)

    actions = full_input_ids
    B, T = full_input_ids.shape
    prompt_mask = torch.zeros((B, T), dtype=torch.float32).cuda()
    for i, prompt_len in enumerate(prompt_lens):
        prompt_mask[i, prompt_len:] = 1.0

    # Get rewards and old policy values
    with torch.no_grad():
        rewards = trainer._compute_rewards(full_input_ids, prompt_mask)

        old_outputs = trainer.old_model(
            full_input_ids, attention_mask=full_attention_mask, output_hidden_states=True
        )
        old_logits = old_outputs.logits
        old_log_probs = trainer._get_log_probs(old_logits, actions)
        old_values = trainer.value_head(old_outputs.hidden_states[-1])

    # Add to buffer
    # Split batch into individual samples and add each one to buffer
    for i in range(full_input_ids.size(0)):
        buffer.add(
            full_input_ids[i],
            full_attention_mask[i],
            prompt_mask[i],
            actions[i],
            rewards[i],
            old_log_probs[i],
            old_values[i],
        )



# Main training loop
if __name__ == "__main__":
    trainer, tokenizer = build_trainer()
    
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
        generate_and_collect(
            prompts=prompts_batch,
            trainer=trainer,
            tokenizer=tokenizer,
            buffer=buffer,
            max_new_tokens=max_new_tokens
        )

        print(f"[Step {step}] Collected {len(buffer.input_ids)} batches")

        # Update the policy every `update_every` batches
        if (step + 1) % update_every == 0:
            print(f"\n[Training Step {step}] Updating policy using {len(buffer.input_ids)} rollouts")

            # Train with PPO
            metrics = trainer.train(*buffer.get())

            print(f"[Train] Loss: {metrics['loss']:.2f}, Policy: {metrics['policy_loss']:.2f}, "
                f"Value: {metrics['value_loss']:.2f}, Entropy: {metrics['entropy']:.4f}, Return: {metrics['mean_return']:.2f}")

            # Update the reference model (old policy)
            trainer.update_old_policy()
            print("Updated old policy")

            # Clear buffer
            buffer.clear()

            # Early stopping if loss explodes
            if metrics["loss"] > 1000:
                print("Loss too high, stopping training")
                break
