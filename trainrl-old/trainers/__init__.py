"""
Trainers package for RL training algorithms.

This package contains base trainer classes and specific implementations
for different RL algorithms like REINFORCE and PPO.
"""

from .base import BaseRLTrainer, RewardFunction
from .reinforce import REINFORCETrainer, compute_advantages
from .ppo import PPOTrainer

__all__ = [
    'BaseRLTrainer',
    'RewardFunction', 
    'REINFORCETrainer',
    'compute_advantages',
    'PPOTrainer'
]