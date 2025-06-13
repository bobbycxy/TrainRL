import torch
import torch.nn.functional as F
import torch.distributed as dist
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all models used in RL training"""
    
    def __init__(self, model, tokenizer, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        
        # Default generation parameters
        self.default_generation_params = {
            'max_new_tokens': 150,
            'temperature': 1.0,
            'top_p': 1.0,
            'do_sample': True,
            'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
    
    def get_model(self):
        """Get the underlying model"""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer"""
        return self.tokenizer
    
    def set_train_mode(self):
        """Set model to training mode"""
        self.model.train()
    
    def set_eval_mode(self):
        """Set model to evaluation mode"""
        self.model.eval()
    
    @abstractmethod
    def generate(self, input_ids, attention_mask=None, **generation_kwargs):
        """Generate text using the model - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through the model - to be implemented by subclasses"""
        pass
    
    def get_generation_params(self, **overrides):
        """Get generation parameters with optional overrides"""
        params = self.default_generation_params.copy()
        params.update(overrides)
        return params


class CausalLMModel(BaseModel):
    """Implementation for causal language models (like GPT, LLaMA, etc.)"""
    
    def __init__(self, model, tokenizer, config=None):
        super().__init__(model, tokenizer, config)
        
        # Override default generation params for causal LMs if needed
        causal_lm_params = {
            'max_new_tokens': getattr(config, 'generation_length', 150) if config else 150,
            'temperature': getattr(config, 'temperature', 1.0) if config else 1.0,
            'do_sample': getattr(config, 'do_sample', True) if config else True,
        }
        self.default_generation_params.update(causal_lm_params)
    
    def generate(self, input_ids, attention_mask=None, **generation_kwargs):
        """Generate text using manual implementation for FSDP compatibility"""
        generation_params = self.get_generation_params(**generation_kwargs)
        max_new_tokens = generation_params['max_new_tokens']
        
        if dist.get_rank() == 0:
            logger.debug(f"Generate: input_ids shape {input_ids.shape}, max_new_tokens {max_new_tokens}")
        
        self.set_eval_mode()
        generated_ids = input_ids.clone()
        batch_size = input_ids.size(0)
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                if dist.get_rank() == 0 and step % 10 == 0:
                    logger.debug(f"Generation step {step}/{max_new_tokens}")
                    
                # Skip if all sequences are finished
                if finished.all():
                    break
                    
                # Get current attention mask
                current_attention_mask = torch.ones_like(generated_ids)
                
                try:
                    outputs = self.forward(generated_ids, attention_mask=current_attention_mask)
                    logits = outputs.logits
                    
                    # Get next token probabilities
                    next_token_logits = logits[:, -1, :]
                    
                    # Apply temperature scaling
                    if generation_params.get('temperature', 1.0) != 1.0:
                        next_token_logits = next_token_logits / generation_params['temperature']
                    
                    # Apply top-p sampling if specified
                    if generation_params.get('top_p', 1.0) < 1.0:
                        next_token_logits = self._apply_top_p_filtering(
                            next_token_logits, generation_params['top_p']
                        )
                    
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Sample next token
                    if generation_params.get('do_sample', True):
                        next_token = torch.multinomial(next_token_probs, 1)
                    else:
                        # Greedy sampling
                        next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)
                    
                    # Append to generated sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Check for EOS tokens in the batch
                    if self.tokenizer.eos_token_id is not None:
                        # Mark sequences as finished if they generate EOS token
                        eos_generated = (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
                        finished = finished | eos_generated
                        
                except Exception as e:
                    if dist.get_rank() == 0:
                        logger.error(f"Error during generation step {step}: {e}")
                    break
        
        if dist.get_rank() == 0:
            logger.debug(f"Generation completed. Final shape: {generated_ids.shape}")
        
        return generated_ids
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through the causal language model"""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def _apply_top_p_filtering(self, logits, top_p):
        """Apply top-p (nucleus) filtering to logits"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Set logits to -inf for tokens to remove
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits


class FSDPCausalLMModel(CausalLMModel):
    """Specialized implementation for FSDP-wrapped causal language models"""
    
    def __init__(self, model, tokenizer, config=None):
        super().__init__(model, tokenizer, config)
        self.is_fsdp = True
    
    def generate(self, input_ids, attention_mask=None, **generation_kwargs):
        """FSDP-compatible generation with additional optimizations"""
        # Add FSDP-specific optimizations if needed
        if dist.get_rank() == 0:
            logger.debug("Using FSDP-optimized generation")
        
        # For now, use the parent implementation
        # In the future, you could add FSDP-specific optimizations here
        return super().generate(input_ids, attention_mask, **generation_kwargs)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """FSDP-compatible forward pass"""
        # FSDP models can be used directly
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def create_model_wrapper(model, tokenizer, config=None, model_type="causal_lm"):
    """Factory function to create appropriate model wrapper"""
    
    # Check if model is FSDP-wrapped
    is_fsdp = hasattr(model, '_fsdp_wrapped_module') or hasattr(model, 'module')
    
    if model_type == "causal_lm":
        if is_fsdp:
            if dist.get_rank() == 0:
                logger.info("Creating FSDP Causal LM model wrapper")
            return FSDPCausalLMModel(model, tokenizer, config)
        else:
            if dist.get_rank() == 0:
                logger.info("Creating standard Causal LM model wrapper")
            return CausalLMModel(model, tokenizer, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")