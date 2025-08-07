import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HFRewardModel(nn.Module):
    def __init__(self, model_name="OpenAssistant/reward-model-deberta-v3-large"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def forward(self, input_ids):
        with torch.no_grad():
            # Decode from token ids to text
            decoded = [self.tokenizer.decode(x, skip_special_tokens=True) for x in input_ids]
            enc = self.tokenizer(decoded, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(input_ids.device) for k, v in enc.items()}
            
            outputs = self.model(**enc)
            scores = outputs.logits.squeeze(-1)  # [B]
            # Expand to sequence length and normalize
            scores = torch.tanh(scores) * 5.0  # Bound rewards to reasonable range
            return scores.unsqueeze(1).expand(-1, input_ids.shape[1])  # [B, T]
