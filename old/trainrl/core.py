# reward_hackers/core.py
"""
Core reward function classes and interfaces.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class RewardFunction(ABC):
    """
    Abstract base class for all reward functions.
    All rewards should return values in [0, 1] range.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.weight = 1.0
    
    @abstractmethod
    def __call__(self, prompt: str, response: str) -> float:
        """
        Compute reward for a prompt-response pair.
        
        Args:
            prompt: Input prompt/instruction
            response: Model's response
            
        Returns:
            Reward score in [0, 1] range
        """
        pass
    
    def __mul__(self, weight: float):
        """Enable reward composition with weights."""
        return WeightedReward(self, weight)
    
    def __add__(self, other):
        """Enable reward combination."""
        return CompositeReward([self, other])
    
    @classmethod
    def load(cls, model_id: str):
        """
        Load a reward function by ID.
        
        Args:
            model_id: Either HuggingFace model ID or rule-based reward name
        """
        if model_id in RULE_BASED_REWARDS:
            return RULE_BASED_REWARDS[model_id]()
        else:
            return HuggingFaceReward(model_id)
    
    def normalize_score(self, score: Union[float, torch.Tensor]) -> float:
        """Normalize score to [0, 1] range."""
        if isinstance(score, torch.Tensor):
            score = score.item()
        return max(0.0, min(1.0, score))


class RuleBasedReward(RewardFunction):
    """
    Base class for rule-based reward functions.
    These don't require neural models and are fast to compute.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        
    @classmethod
    def create(cls, reward_type: str, **kwargs):
        """Factory method to create rule-based rewards."""
        if reward_type == "math":
            from .rewards import MathReward
            return MathReward(**kwargs)
        elif reward_type == "code":
            from .rewards import CodeReward
            return CodeReward(**kwargs)
        else:
            raise ValueError(f"Unknown rule-based reward type: {reward_type}")


class HuggingFaceReward(RewardFunction):
    """
    Wrapper for HuggingFace reward models.
    """
    
    def __init__(self, model_id: str, device: str = "auto"):
        super().__init__(f"hf_{model_id.split('/')[-1]}")
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded reward model: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise
    
    def __call__(self, prompt: str, response: str) -> float:
        """Compute reward using HuggingFace model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Combine prompt and response
        text = f"{prompt}\n{response}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Assume single output logit or probability
            if outputs.logits.shape[-1] == 1:
                score = torch.sigmoid(outputs.logits).item()
            else:
                # Multi-class case - take positive class probability
                score = torch.softmax(outputs.logits, dim=-1)[0, -1].item()
        
        return self.normalize_score(score)


class WeightedReward(RewardFunction):
    """Applies a weight to another reward function."""
    
    def __init__(self, reward_fn: RewardFunction, weight: float):
        super().__init__(f"weighted_{reward_fn.name}")
        self.reward_fn = reward_fn
        self.weight = weight
    
    def __call__(self, prompt: str, response: str) -> float:
        base_score = self.reward_fn(prompt, response)
        return self.normalize_score(base_score * self.weight)


class CompositeReward(RewardFunction):
    """Combines multiple reward functions."""
    
    def __init__(self, reward_fns: list):
        names = "_".join([rf.name for rf in reward_fns])
        super().__init__(f"composite_{names}")
        self.reward_fns = reward_fns
    
    def __call__(self, prompt: str, response: str) -> float:
        total_score = sum(rf(prompt, response) for rf in self.reward_fns)
        return self.normalize_score(total_score / len(self.reward_fns))


# Registry of available rule-based rewards
RULE_BASED_REWARDS = {
    "math_basic": lambda: RuleBasedReward.create("math"),
    "code_basic": lambda: RuleBasedReward.create("code"),
}