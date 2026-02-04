import torch
import math
def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    mean = torch.mean(X, dim=-1, keepdim=True)
    var = torch.var(X, dim=-1, keepdim=True, unbiased=False)

    normalized = (X - mean) / torch.sqrt(var + epsilon)
    return gamma * normalized + beta