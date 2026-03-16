import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    scores = torch.tensor(scores)
    scores = torch.exp(scores)
    total = torch.sum(scores)

    return (scores / total).tolist()
