# examples/math_training_example.py
"""
Example: Training a small language model on math problems using RewardHackers.
Run this file directly: python examples/math_training_example.py
"""

import sys
import os
import logging

# Add the parent directory to Python path so we can import reward_hackers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from the trainrl directory
from trainrl.core import RewardFunction
from trainrl.trainer import REINFORCETrainer, TrainingConfig
from trainrl.rewards import MathReward, LengthReward

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("=== RewardHackers Math Training Example ===")
    
    # Example math problems (we'll use one answer for simplicity)
    math_problems = [
        "What is 15 + 27?",
        "Calculate 8 × 9", 
        "What is 144 ÷ 12?",
        "What is 25% of 80?",
        "Find the area of a rectangle with length 6 and width 4",
    ]
    
    # For this demo, we'll pretend all answers should be "42" 
    # (in real use, you'd have different correct answers per problem)
    
    # Create evaluation prompts
    eval_problems = [
        "What is 20 + 22?",  # This actually equals 42!
        "Calculate 7 × 6",   # This equals 42 too!
    ]
    
    # 1. Create a math reward function
    print("Creating math reward function...")
    math_reward = MathReward(correct_answer="42")
    
    # 2. Create a length reward (optional - encourages reasonable length responses)
    length_reward = LengthReward(target_length=100, tolerance=30)
    
    # 3. Combine rewards (80% math correctness, 20% length preference)
    combined_reward = math_reward * 0.8 + length_reward * 0.2
    print(f"Using combined reward function")
    
    # 4. Set up training configuration (very small for demo)
    config = TrainingConfig(
        learning_rate=1e-5,
        batch_size=1,      # Very small batch for demo
        num_epochs=1,      # Just 1 epoch for quick demo
        max_length=128,    # Short responses
        logging_steps=1,   # Log every step
        save_steps=3,      # Evaluate every 3 steps
        gradient_clip=1.0
    )
    
    # 5. Initialize trainer with a small model (GPT-2 for demo)
    print("Initializing REINFORCE trainer...")
    print("Note: This will download GPT-2 model if not already cached")
    
    try:
        trainer = REINFORCETrainer(
            model_name="gpt2",  # Small 124M parameter model
            reward_function=combined_reward,
            config=config
        )
        
        # 6. Test the reward function first
        print("\n=== Testing Reward Function ===")
        test_prompt = "What is 20 + 22?"
        test_good_response = "<think>20 + 22 = 42</think>\nThe answer is \\boxed{42}"
        test_bad_response = "I don't know the answer"
        
        good_score = combined_reward(test_prompt, test_good_response)
        bad_score = combined_reward(test_prompt, test_bad_response)
        
        print(f"Good response score: {good_score:.3f}")
        print(f"Bad response score: {bad_score:.3f}")
        
        # 7. Generate a sample response before training
        print("\n=== Sample Generation Before Training ===")
        sample_response, _, _ = trainer.generate_response(eval_problems[0], do_sample=False)
        sample_reward = combined_reward(eval_problems[0], sample_response)
        print(f"Prompt: {eval_problems[0]}")
        print(f"Response: {sample_response}")
        print(f"Reward: {sample_reward:.3f}")
        
        # 8. Train the model (just a few steps for demo)
        print("\n=== Starting Training ===")
        print("Note: This is a minimal demo - real training needs more data and epochs")
        
        history = trainer.train(
            train_prompts=math_problems[:3],  # Use subset for quick demo
            eval_prompts=eval_problems
        )
        
        # 9. Generate sample after training
        print("\n=== Sample Generation After Training ===")
        sample_response_after, _, _ = trainer.generate_response(eval_problems[0], do_sample=False)
        sample_reward_after = combined_reward(eval_problems[0], sample_response_after)
        print(f"Prompt: {eval_problems[0]}")
        print(f"Response: {sample_response_after}")
        print(f"Reward: {sample_reward_after:.3f}")
        
        # 10. Show training history
        print("\n=== Training Results ===")
        if history['loss']:
            print(f"Final loss: {history['loss'][-1]:.3f}")
            print(f"Final reward: {history['avg_reward'][-1]:.3f}")
        
        print("\n=== Demo Complete! ===")
        print("This was a minimal demo. For real training:")
        print("- Use more diverse training data")
        print("- Train for more epochs")
        print("- Use larger batch sizes")
        print("- Implement proper evaluation metrics")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch transformers accelerate")


if __name__ == "__main__":
    main()