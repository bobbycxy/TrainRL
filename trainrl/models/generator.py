"""
This code will build the generators that will be used to generate text.
"""

import torch

class GeneratorInterface(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def generate(self, x):
        raise NotImplementedError
    
class OpenRouterGenerator(GeneratorInterface):
    def __init__(self, model, generate_cfg, device="cuda"):
        super().__init__(model)
        self.generate_cfg = generate_cfg
        self.device = device
        
    

class StandardGenerator(GeneratorInterface):
    def __init__(self, model, generate_cfg, device="cuda"):
        super().__init__(model)
        self.generate_cfg = generate_cfg
        self.device = device
        self.model = model.to(device)
    
    @torch.no_grad()
    def generate(self, input_text, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text using the model with support for greedy and top-k sampling.
        """
        # Tokenize the input text
        input_ids = self.model.embedding_model.tokenize_input(input_text, add_eot=False, truncate_from="right")
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)

        for _ in range(max_new_tokens):
            # Get model predictions (logits)
            logits = self.model.inference(input_ids)
            
            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                # Retain only top-k logits
                top_k_logits, _ = torch.topk(logits, k=top_k, dim=-1)
                min_top_k_logit = top_k_logits[-1]  # Smallest value in top-k
                logits[logits < min_top_k_logit] = -float('inf')  # Mask values outside top-k

            # Apply sampling
            if top_k == 1:  # Greedy sampling case
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            else:  # Top-k sampling case
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).unsqueeze(0)

            # Check for end-of-text token and stop if encountered
            if next_token.item() == self.model.embedding_model.eot_token_id:
                break
            
            # Append the next token to the input sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Decode the generated tokens back to text
        return self.model.embedding_model.decode(input_ids.tolist())

    
GeneratorDict = {
    "standard": lambda model, generate_cfg, device: StandardGenerator(model, generate_cfg, device)
}

def build_generator(model, generate_cfg, device="cuda"):
    """
    Build the generator
    """
    generator_type = generate_cfg.get("generator_type", "standard")
    return GeneratorDict[generator_type](model, generate_cfg, device)