"""
Reward Reinforcement Learning (trainrl): A plug-and-play library of reward functions for LLM RL training.
"""

from .core import RewardFunction, RuleBasedReward, HuggingFaceReward
from .trainer import REINFORCETrainer
from .rewards import MathReward, CodeReward, HelpfulnessReward

__version__ = "0.1.0"
__all__ = [
    "RewardFunction", 
    "RuleBasedReward", 
    "HuggingFaceReward",
    "REINFORCETrainer",
    "MathReward",
    "CodeReward", 
    "HelpfulnessReward"
]