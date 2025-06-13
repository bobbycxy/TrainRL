# trainrl/data.py
"""
Data loading utilities for training.
"""

import json
import random
from typing import List, Dict, Tuple
from pathlib import Path

def load_math_dataset(dataset_name: str = "gsm8k", num_samples: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """
    Load math datasets for training.
    
    Args:
        dataset_name: Name of dataset ("gsm8k", "math", or "synthetic")
        num_samples: Number of training samples to load
        
    Returns:
        Tuple of (train_data, test_data)
    """
    if dataset_name == "gsm8k":
        return load_gsm8k_data(num_samples)
    elif dataset_name == "math":
        return load_hendrycks_math(num_samples)
    elif dataset_name == "synthetic":
        return load_synthetic_math_data(num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_gsm8k_data(num_samples: int = 100):
    """
    Load GSM8K dataset for math training - Fixed version.
    """
    try:
        from datasets import load_dataset
        
        logger.info("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main")
        
        # Extract training examples - using list indexing instead of iteration
        train_data = []
        train_dataset = dataset["train"]
        
        for i in range(min(num_samples, len(train_dataset))):
            # Access by index
            question = train_dataset[i]["question"]
            answer_text = train_dataset[i]["answer"]
            
            # Extract the final numerical answer
            answer = answer_text.split("####")[-1].strip()
            
            train_data.append({
                "prompt": f"Solve this math problem: {question}",
                "answer": answer
            })
        
        # Extract test examples  
        test_data = []
        test_dataset = dataset["test"]
        
        for i in range(min(20, len(test_dataset))):
            question = test_dataset[i]["question"]
            answer_text = test_dataset[i]["answer"]
            answer = answer_text.split("####")[-1].strip()
            
            test_data.append({
                "prompt": f"Solve this math problem: {question}",
                "answer": answer
            })
        
        logger.info(f"Loaded {len(train_data)} training and {len(test_data)} test examples")
        
        # Debug: print first example
        logger.info(f"First training example: {train_data[0]}")
        
        return train_data, test_data
        
    except Exception as e:
        logger.warning(f"Failed to load GSM8K dataset: {e}")
        logger.warning("Falling back to synthetic data")
        return load_synthetic_math_data(num_samples)


def load_hendrycks_math(num_samples: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """
    Load Hendrycks MATH dataset - competition math problems.
    More challenging than GSM8K.
    """
    try:
        from datasets import load_dataset
        
        print("Loading Hendrycks MATH dataset...")
        dataset = load_dataset("hendrycks/competition_math")
        
        train_data = []
        for i, example in enumerate(dataset["train"]):
            if i >= num_samples:
                break
            
            problem = example["problem"].strip()
            solution = example["solution"].strip()
            
            train_data.append({
                "prompt": f"Solve this math competition problem:\n{problem}",
                "answer": solution,
                "level": example["level"],
                "type": example["type"]
            })
        
        test_data = []
        for i, example in enumerate(dataset["test"][:20]):
            problem = example["problem"].strip()
            solution = example["solution"].strip()
            
            test_data.append({
                "prompt": f"Solve this math competition problem:\n{problem}",
                "answer": solution,
                "level": example["level"],
                "type": example["type"]
            })
        
        print(f"Loaded MATH: {len(train_data)} train, {len(test_data)} test examples")
        return train_data, test_data
        
    except ImportError:
        print("HuggingFace datasets not available, falling back to synthetic data")
        return load_synthetic_math_data(num_samples)


def load_synthetic_math_data(num_samples: int = 100):
    """
    Generate synthetic math problems as fallback - Enhanced version.
    """
    import random
    
    logger.info("Generating synthetic math data...")
    
    train_data = []
    test_data = []
    
    # Enhanced problem templates
    templates = [
        ("What is {a} + {b}?", lambda a, b: a + b, (1, 50), (1, 50)),
        ("Calculate {a} - {b}", lambda a, b: a - b, (10, 100), (1, 30)),
        ("What is {a} × {b}?", lambda a, b: a * b, (2, 15), (2, 12)),
        ("What is {a} ÷ {b}?", lambda a, b: a // b, (10, 100), (2, 10)),
        ("Sarah has {a} apples. She gives away {b}. How many are left?", lambda a, b: a - b, (10, 50), (1, 20)),
        ("A box has {a} items. With {b} boxes, how many items total?", lambda a, b: a * b, (5, 20), (2, 8)),
        ("What is {a}% of {b}?", lambda a, b: (a * b) // 100, (10, 50), (20, 200)),
    ]
    
    for i in range(num_samples):
        template, func, (a_min, a_max), (b_min, b_max) = random.choice(templates)
        
        a = random.randint(a_min, a_max)
        b = random.randint(b_min, b_max)
        
        # Ensure valid division
        if "÷" in template and b == 0:
            b = random.randint(1, 10)
        if "÷" in template:
            a = a - (a % b)  # Make it divide evenly
        
        question = template.format(a=a, b=b)
        answer = str(func(a, b))
        
        problem = {
            "prompt": f"Solve this math problem: {question}",
            "answer": answer
        }
        
        # 80/20 split
        if i < num_samples * 0.8:
            train_data.append(problem)
        else:
            test_data.append(problem)
    
    logger.info(f"Generated {len(train_data)} training and {len(test_data)} test examples")
    logger.info(f"Sample problem: {train_data[0]}")
    
    return train_data, test_data
    

def create_evaluation_prompts(difficulty: str = "easy") -> List[str]:
    """
    Create a set of evaluation prompts for monitoring.
    """
    if difficulty == "easy":
        return [
            "What is 15 + 27?",
            "Calculate 8 × 9",
            "What is 144 ÷ 12?",
            "Sarah has 20 cookies. She eats 3. How many are left?",
            "What is 25% of 80?"
        ]
    elif difficulty == "medium":
        return [
            "Solve: 2x + 5 = 13, find x",
            "What is the area of a rectangle with length 12 and width 8?",
            "If a train travels 60 mph for 2.5 hours, how far does it go?",
            "What is the square root of 144?",
            "Convert 3/4 to a percentage"
        ]
    elif difficulty == "hard":
        return [
            "Solve the quadratic equation: x² - 5x + 6 = 0",
            "What is the derivative of x³ + 2x² - 5x + 1?",
            "Find the area of a circle with radius 7",
            "Solve: log₂(8x) = 5",
            "What is the sum of the first 10 prime numbers?"
        ]
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")


def save_training_data(data: List[Dict], filepath: str):
    """Save training data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} examples to {filepath}")


def load_training_data(filepath: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {filepath}")
    return data