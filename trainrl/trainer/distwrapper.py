from trainer.utils import get_transformer_layer_classes
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch


class BaseDistributedWrapper:
    def __init__(self, device, model, rank):
        self.original_model = model
        self.rank = rank
        self.device = device
        self.wrapped_model = self._wrap_model()
    
    def _wrap_model(self):
        """Override in subclasses to implement specific wrapping logic."""
        raise NotImplementedError("Subclasses must implement _wrap_model")
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper callable like a PyTorch model."""
        return self.wrapped_model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate all attribute access to the wrapped model."""
        return getattr(self.wrapped_model, name)


class FSDPWrapper(BaseDistributedWrapper):
    def _wrap_model(self):
        print(f"[Rank {self.rank}] Using FSDP...")
        transformer_layer_classes = get_transformer_layer_classes(self.original_model)
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_classes
        )

        # Mixed precision configuration for better performance
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            reduce_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            buffer_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

        return FSDP(
            self.original_model,
            device_id=self.rank,
            auto_wrap_policy=auto_wrap_policy,
            # mixed_precision=mixed_precision,
            sync_module_states=True,
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=False,
        )
    
    def _is_fsdp(self):
        """Check if this is an FSDP wrapper"""
        return True
    
    def _get_model_dtype(self):
        """Get the dtype of the model parameters"""
        for param in self.wrapped_model.parameters():
            return param.dtype
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    def _ensure_input_dtype_compatibility(self, *args, **kwargs):
        """Ensure input tensors are compatible with model dtype"""
        model_dtype = self._get_model_dtype()
        
        def convert_tensor(tensor):
            if isinstance(tensor, torch.Tensor):
                # Keep integer tensors (like input_ids, attention_mask) as-is for embeddings
                if tensor.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    return tensor
                # Convert float tensors to model dtype
                elif tensor.dtype in [torch.float32, torch.float64]:
                    return tensor.to(model_dtype)
            return tensor
        
        # Convert args
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(convert_tensor(arg))
            else:
                new_args.append(arg)
        
        # Convert kwargs
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                new_kwargs[key] = convert_tensor(value)
            else:
                new_kwargs[key] = value
                
        return tuple(new_args), new_kwargs
    
    def _safe_fsdp_call(self, method_name, *args, **kwargs):
        """Safely call FSDP methods avoiding nested summon_full_params"""
        try:
            # First try without summon_full_params (parameters might already be unsharded)
            if hasattr(self.wrapped_model, method_name):
                args, kwargs = self._ensure_input_dtype_compatibility(*args, **kwargs)
                return getattr(self.wrapped_model, method_name)(*args, **kwargs)
            else:
                args, kwargs = self._ensure_input_dtype_compatibility(*args, **kwargs)
                return self.wrapped_model(*args, **kwargs)
        except RuntimeError as e:
            if "weight' must be 2-D" in str(e) or "expected mat1 and mat2" in str(e):
                # Need to use summon_full_params
                with FSDP.summon_full_params(self.wrapped_model, writeback=False):
                    if hasattr(self.wrapped_model, method_name):
                        args, kwargs = self._ensure_input_dtype_compatibility(*args, **kwargs)
                        return getattr(self.wrapped_model, method_name)(*args, **kwargs)
                    else:
                        args, kwargs = self._ensure_input_dtype_compatibility(*args, **kwargs)
                        return self.wrapped_model(*args, **kwargs)
            else:
                raise e
    
    def generate_with_fsdp(self, *args, **kwargs):
        """Generate method that works with FSDP"""
        with torch.no_grad():
            return self._safe_fsdp_call('generate', *args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Override generate to use safe FSDP call"""
        return self.generate_with_fsdp(*args, **kwargs)
    
    def forward_with_fsdp(self, *args, **kwargs):
        """Forward pass that works with FSDP"""
        return self._safe_fsdp_call(None, *args, **kwargs)  # None means call the model directly
    
    def generate_hidden_states(self, *args, **kwargs):
        """Generate hidden states with FSDP context"""
        return self._safe_fsdp_call('generate_hidden_states', *args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper callable - remove automatic FSDP context to avoid nesting"""
        # Convert dtype but don't automatically add FSDP context
        args, kwargs = self._ensure_input_dtype_compatibility(*args, **kwargs)
        return self.wrapped_model(*args, **kwargs)
    
    def manual_generate(self, input_ids, attention_mask, max_new_tokens=32, 
                       temperature=0.7, top_k=50, top_p=0.9, pad_token_id=None):
        """Manual generation without FSDP context management"""
        import torch.nn.functional as F
        
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Ensure input tensors have compatible dtype
        model_dtype = self._get_model_dtype()
        if attention_mask.dtype == torch.float32 and model_dtype in [torch.bfloat16, torch.float16]:
            attention_mask = attention_mask.to(model_dtype)
        
        # Start with the input
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.wrapped_model(
                    input_ids=generated if past_key_values is None else generated[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)
                
                # Check for EOS tokens
                if pad_token_id is not None and (next_token == pad_token_id).all():
                    break
        
        return generated


class DDPWrapper(BaseDistributedWrapper):
    def _wrap_model(self):
        print(f"[Rank {self.rank}] Using DDP...")
        from torch.nn.parallel import DistributedDataParallel as DDP 
        return DDP(self.original_model, device_ids=[self.rank])
    
    def _is_fsdp(self):
        """Check if this is an FSDP wrapper"""
        return False
    
    def generate_with_fsdp(self, *args, **kwargs):
        """For compatibility - DDP doesn't need special generation handling"""
        return self.wrapped_model.generate(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Regular generate for DDP"""
        return self.wrapped_model.generate(*args, **kwargs)
    
    def forward_with_fsdp(self, *args, **kwargs):
        """For compatibility - regular forward for DDP"""
        return self.wrapped_model(*args, **kwargs)
    
    def generate_hidden_states(self, *args, **kwargs):
        """Regular generate_hidden_states for DDP"""
        return self.wrapped_model.generate_hidden_states(*args, **kwargs)
    
    def manual_generate(self, *args, **kwargs):
        """For compatibility - just use regular generate for DDP"""
        return self.generate_with_fsdp(*args, **kwargs)


DISTWRAPPER_DICT = {
    "fsdp": lambda device, model, rank: FSDPWrapper(device, model, rank),
    "ddp":  lambda device, model, rank: DDPWrapper(device, model, rank)
}


def build_distwrapper(cfg, device, model, rank):
    return DISTWRAPPER_DICT[cfg["trainer"].get("parallelism").lower()](device, model, rank)