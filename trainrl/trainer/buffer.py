from torch.nn.utils.rnn import pad_sequence

class RolloutBuffer:
    def __init__(self):
        self.input_ids = []
        self.attn_masks = []
        self.prompt_masks = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []
        self.old_values = []

    def add(self, input_ids, attn_mask, prompt_mask, actions, rewards, log_probs, values):
        self.input_ids.append(input_ids)
        self.attn_masks.append(attn_mask)
        self.prompt_masks.append(prompt_mask)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.old_log_probs.append(log_probs)
        self.old_values.append(values)

    def get(self):
        # Pad sequences to the same length
        def pad(tensors):
            return pad_sequence(tensors, batch_first=True, padding_value=0)

        return (
            pad(self.input_ids),
            pad(self.attn_masks),
            pad(self.prompt_masks),
            pad(self.actions),
            pad(self.rewards),
            pad(self.old_log_probs),
            pad(self.old_values),
        )

    def clear(self):
        self.__init__()