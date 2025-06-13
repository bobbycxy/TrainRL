from dataclasses import dataclass
from enum import Enum

class RLAlgorithm(Enum):
    """Enum for RL algorithm selection"""
    REINFORCE = "reinforce"
    PPO = "ppo"


@dataclass
class RLConfig:
    """Configuration for RL training"""
    model_name: str = "openai-community/gpt2-large"
    dataset_name: str = "gsm8k"
    max_length: int = 256  # Reduced from 512
    batch_size: int = 2    # Reduced from 4 for debugging
    learning_rate: float = 1e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # REINFORCE specific
    baseline_coef: float = 0.5  # Weight for value function baseline
    
    # PPO specific
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Generation
    generation_length: int = 128  # Much smaller for debugging
    
    # Algorithm selection
    rl_algorithm: RLAlgorithm = RLAlgorithm.REINFORCE  # Toggle between REINFORCE and PPO