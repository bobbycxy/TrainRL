# TrainRL (WIP)

**TrainRL** is a reinforcement learning playground built to experiment with fine-tuning language models using REINFORCE and PPO — especially focused on learning by doing.

Right now, it trains models like GPT-2 using math questions (GSM8K) and rewards them based on how well they solve the problem.

The goal: give users a clean, hackable base to try new reward functions, new policy algorithms, or attach their own game environments.



## What Works Now

- ✅ Hugging Face LLMs (`AutoModelForCausalLM`)
- ✅ Fully Sharded Data Parallel (FSDP) training with PyTorch
- ✅ Custom token-level autoregressive `.generate()` implementation (FSDP-safe)
- ✅ Simple REINFORCE and PPO trainers
- ✅ Reward function that scores math answers using regex
- ✅ Dataset: GSM8K, with dataloader and prompt formatting
- ✅ Works with `torchrun` on multi-GPU setups

## Getting Started

```bash
torchrun --nproc_per_node=4 sandbox2.py
```
