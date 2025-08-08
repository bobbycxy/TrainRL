import torch.nn.functional as F

def get_log_probs(logits, actions):
    # Add numerical stability
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    return action_log_probs