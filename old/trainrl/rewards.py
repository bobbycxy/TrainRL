# reward_hackers/rewards.py
"""
Specific reward function implementations.
"""

import re
import math
import subprocess
import tempfile
import os
from typing import List, Optional
from .core import RuleBasedReward, HuggingFaceReward


class MathReward(RuleBasedReward):
    """
    Rule-based reward for mathematical problems.
    Based on DeepSeek-Math approach with accuracy, length, and format components.
    """
    
    def __init__(self, 
                 correct_answer: str,
                 weight_accuracy: float = 0.6,
                 weight_cosine: float = 0.3, 
                 weight_format: float = 0.1):
        super().__init__("math_reward")
        self.correct_answer = str(correct_answer)
        self.weights = (weight_accuracy, weight_cosine, weight_format)
    
    def accuracy_reward(self, response: str) -> float:
        """Binary reward for correctness using \\boxed{} format."""
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
        if not boxed_match:
            return 0.0
        
        predicted_answer = boxed_match.group(1).strip()
        return 1.0 if predicted_answer == self.correct_answer else 0.0
    
    def cosine_reward(self, response: str, accuracy: float) -> float:
        """Scale accuracy by response length using cosine schedule."""
        length = len(response)
        
        if accuracy == 0:
            # Longer incorrect solutions penalized less severely
            length_penalty = 1 - math.cos(min(length / 1000, math.pi/2))
            return -0.1 * length_penalty
        else:
            # Shorter correct solutions get higher rewards
            length_factor = math.cos(min(length / 500, math.pi/2))
            return accuracy * length_factor
    
    def format_reward(self, response: str) -> float:
        """Reward for proper <think> </think> structure."""
        has_think_tags = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
        return 0.2 if has_think_tags else 0.0
    
    def __call__(self, prompt: str, response: str) -> float:
        accuracy = self.accuracy_reward(response)
        cosine = self.cosine_reward(response, accuracy)
        format_score = self.format_reward(response)
        
        total = (self.weights[0] * accuracy + 
                self.weights[1] * cosine + 
                self.weights[2] * format_score)
        
        return self.normalize_score(total)


class CodeReward(RuleBasedReward):
    """
    Rule-based reward for code generation tasks.
    Evaluates correctness, style, and efficiency.
    """
    
    def __init__(self, 
                 test_cases: List[dict],
                 weight_correctness: float = 0.7,
                 weight_style: float = 0.2,
                 weight_efficiency: float = 0.1):
        super().__init__("code_reward")
        self.test_cases = test_cases
        self.weights = (weight_correctness, weight_style, weight_efficiency)
    
    def extract_code(self, response: str) -> str:
        """Extract Python code from response."""
        # Look for code blocks
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Fallback: assume entire response is code
        return response.strip()
    
    def correctness_reward(self, code: str) -> float:
        """Test code correctness using provided test cases."""
        if not self.test_cases:
            return 1.0
        
        passed = 0
        total = len(self.test_cases)
        
        for test_case in self.test_cases:
            try:
                # Create temporary file with code
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    f.write('\n\n')
                    f.write(f"result = {test_case['call']}")
                    f.write(f"\nprint('RESULT:', result)")
                    temp_file = f.name
                
                # Execute and check result
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if f"RESULT: {test_case['expected']}" in output:
                        passed += 1
                
                os.unlink(temp_file)
                
            except Exception:
                continue
        
        return passed / total if total > 0 else 0.0
    
    def style_reward(self, code: str) -> float:
        """Simple style checks."""
        score = 0.0
        
        # Has docstring
        if '"""' in code or "'''" in code:
            score += 0.3
        
        # Has comments
        if '#' in code:
            score += 0.3
        
        # Reasonable variable names (not single letters for non-loop vars)
        lines = code.split('\n')
        good_names = 0
        total_assignments = 0
        
        for line in lines:
            if '=' in line and not line.strip().startswith('#'):
                total_assignments += 1
                var_name = line.split('=')[0].strip()
                if len(var_name) > 1 or var_name in 'ijk':  # Allow i,j,k for loops
                    good_names += 1
        
        if total_assignments > 0:
            score += 0.4 * (good_names / total_assignments)
        
        return min(1.0, score)
    
    def efficiency_reward(self, code: str) -> float:
        """Basic efficiency heuristics."""
        # Penalize obvious inefficiencies
        score = 1.0
        
        # Nested loops (potential O(n²))
        nested_loops = len(re.findall(r'for.*:\s*.*for.*:', code, re.DOTALL))
        score -= 0.2 * nested_loops
        
        # Multiple passes through same data
        list_iterations = len(re.findall(r'for.*in.*list', code))
        if list_iterations > 2:
            score -= 0.1 * (list_iterations - 2)
        
        return max(0.0, score)
    
    def __call__(self, prompt: str, response: str) -> float:
        code = self.extract_code(response)
        
        correctness = self.correctness_reward(code)
        style = self.style_reward(code)
        efficiency = self.efficiency_reward(code)
        
        total = (self.weights[0] * correctness +
                self.weights[1] * style +
                self.weights[2] * efficiency)
        
        return self.normalize_score(total)


class HelpfulnessReward(HuggingFaceReward):
    """
    Helpfulness reward using OpenAssistant's reward model.
    """
    
    def __init__(self):
        super().__init__("OpenAssistant/reward-model-deberta-v3-large-v2")


class SafetyReward(HuggingFaceReward):
    """
    Safety reward using PKU-Alignment's safety model.
    """
    
    def __init__(self):
        # Note: This is a placeholder - replace with actual safety model
        super().__init__("PKU-Alignment/beaver-7b-v1.0-reward")


class LengthReward(RuleBasedReward):
    """
    Simple length-based reward - encourages appropriate response length.
    """
    
    def __init__(self, target_length: int = 200, tolerance: int = 50):
        super().__init__("length_reward")
        self.target_length = target_length
        self.tolerance = tolerance
    
    def __call__(self, prompt: str, response: str) -> float:
        length = len(response)
        diff = abs(length - self.target_length)
        
        if diff <= self.tolerance:
            return 1.0
        else:
            # Exponential decay based on how far from target
            penalty = diff / self.target_length
            return self.normalize_score(math.exp(-penalty))


class FormatReward(RuleBasedReward):
    """
    Rewards proper formatting (markdown, structure, etc.).
    """
    
    def __init__(self, required_format: str = "markdown"):
        super().__init__("format_reward")
        self.required_format = required_format
    
    def __call__(self, prompt: str, response: str) -> float:
        score = 0.0
        
        if self.required_format == "markdown":
            # Check for markdown elements
            if re.search(r'#{1,6}\s', response):  # Headers
                score += 0.3
            if re.search(r'\*\*.*?\*\*', response):  # Bold
                score += 0.2
            if re.search(r'\*.*?\*', response):  # Italics
                score += 0.2
            if re.search(r'```.*?```', response, re.DOTALL):  # Code blocks
                score += 0.3
        
        return self.normalize_score(score)