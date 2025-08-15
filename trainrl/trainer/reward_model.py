import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HFRewardModel(nn.Module):
    def __init__(self, model_name="OpenAssistant/reward-model-deberta-v3-large"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def forward(self, input_texts):
        """
        Forward pass for reward model.
        
        Args:
            input_texts: List of strings (prompt + completion texts)
            
        Returns:
            scores: [B] tensor of reward scores
        """
        with torch.no_grad():
            # Tokenize the input texts using the reward model's tokenizer
            enc = self.tokenizer(
                input_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Move tokenized inputs to the same device as the model
            device = next(self.model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            
            # Get reward scores from the model
            outputs = self.model(**enc)
            scores = outputs.logits.squeeze(-1)  # [B]
            
            # Bound rewards to reasonable range
            scores = torch.tanh(scores) * 5.0
            
            return scores  # Return [B] tensor, expansion handled in caller