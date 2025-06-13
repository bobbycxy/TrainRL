import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial


def generate_with_model(model, tokenizer, input_ids, attention_mask, max_new_tokens=150):
    """Generate text using the model (manual implementation for FSDP compatibility)"""
    if dist.get_rank() == 0:
        print(f"Generate: input_ids shape {input_ids.shape}, max_new_tokens {max_new_tokens}")
    
    model.eval()
    generated_ids = input_ids.clone()
    batch_size = input_ids.size(0)
    
    # Track which sequences are finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            if dist.get_rank() == 0 and step % 5 == 0:
                print(f"Generation step {step}/{max_new_tokens}")
                
            # Skip if all sequences are finished
            if finished.all():
                break
                
            # Get current attention mask
            current_attention_mask = torch.ones_like(generated_ids)
            
            try:
                outputs = model(generated_ids, attention_mask=current_attention_mask)
                logits = outputs.logits
                
                # Get next token probabilities
                next_token_logits = logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token (you could also use greedy or beam search)
                next_token = torch.multinomial(next_token_probs, 1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Check for EOS tokens in the batch
                if tokenizer.eos_token_id is not None:
                    # Mark sequences as finished if they generate EOS token
                    eos_generated = (next_token.squeeze(-1) == tokenizer.eos_token_id)
                    finished = finished | eos_generated
                    
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"Error during generation step {step}: {e}")
                break
    
    if dist.get_rank() == 0:
        print(f"Generation completed. Final shape: {generated_ids.shape}")
    
    return generated_ids

class BaseModel:
    def __init__(self, model_name: str, fsdp: bool = True, min_params: int = int(1e6)):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()

        if fsdp:
            wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_params)
            self.model = FSDP(
                self.model,
                auto_wrap_policy=wrap_policy,
                device_id=torch.cuda.current_device()
            )

    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 2):
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            blocks = self.model.transformer.h
            for block in blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True

        if hasattr(self.model, "lm_head"):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
        if hasattr(self.model.transformer, "ln_f"):
            for param in self.model.transformer.ln_f.parameters():
                param.requires_grad = True

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def generate(self, input_ids, attention_mask, max_new_tokens=150):
        return generate_with_model(self.model, self.tokenizer, input_ids, attention_mask, max_new_tokens)
