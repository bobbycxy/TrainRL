import torch
from torch.nn.utils.rnn import pad_sequence

class RolloutBuffer:
    def __init__(self):
        self.actions = []  # This contains the full sequence (prompt + completion tokens)
        self.attn_masks = []
        self.prompt_masks = []
        self.rewards = []
        self.old_log_probs = []
        self.old_values = []

    def add(self, actions, attn_mask, prompt_mask, rewards, log_probs, values):
        # Only store actions (which are the full sequence tokens)
        self.actions.append(actions.cpu() if hasattr(actions, 'cpu') else actions)
        self.attn_masks.append(attn_mask.cpu() if hasattr(attn_mask, 'cpu') else attn_mask)
        self.prompt_masks.append(prompt_mask.cpu() if hasattr(prompt_mask, 'cpu') else prompt_mask)
        self.rewards.append(rewards.cpu() if hasattr(rewards, 'cpu') else rewards)
        self.old_log_probs.append(log_probs.cpu() if hasattr(log_probs, 'cpu') else log_probs)
        self.old_values.append(values.cpu() if hasattr(values, 'cpu') else values)

    def get(self, target_device=None):
        if not self.actions:
            return None
        
        def pad(tensors):
            padded = pad_sequence(tensors, batch_first=True, padding_value=0)
            if target_device is not None:
                padded = padded.to(target_device)
            return padded

        actions = pad(self.actions)
        attn_masks = pad(self.attn_masks) 
        prompt_masks = pad(self.prompt_masks)
        rewards = pad(self.rewards)
        old_log_probs = pad(self.old_log_probs)
        old_values = pad(self.old_values)

        return (
            actions,        # The full token sequence
            attn_masks,
            prompt_masks,
            rewards,
            old_log_probs,
            old_values,
        )

    def clear(self):
        self.__init__()