import torch

def cross_entropy(inputs, targets):
    center = inputs.max(dim = -1, keepdim = True).values
    shifted = inputs - center
    log_normalizer = torch.log(torch.exp(shifted).sum(dim = -1))
    chosen = shifted.gather(dim = -1, index = targets.unsqueeze(-1))
    chosen = chosen.squeeze(-1)
    loss = log_normalizer - chosen
    return loss.mean()
        