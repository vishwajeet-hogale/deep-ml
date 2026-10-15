import torch
import torch.nn.functional as F


def cross_entropy(logits, targets):
    num_classes = logits.shape[-1]
    one_hot = F.one_hot(targets, num_classes=num_classes)

    logits_max, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - logits_max


    exp_logits = torch.exp(logits)
    exp_logits_sum = exp_logits.sum(dim=-1, keepdim=True)

    sm = exp_logits / exp_logits_sum

    log_sm = torch.log(sm)

    loss = (-1 * one_hot * log_sm).sum(dim=-1).mean()

    return loss

