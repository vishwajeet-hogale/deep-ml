import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    scores = torch.tensor(scores, dtype = torch.float32)
    scores -= torch.max(scores)
    
    exp_scores = torch.exp(scores)


    sum_exp_scores = exp_scores.sum()

    return torch.log(exp_scores / (sum_exp_scores + 1e-7))