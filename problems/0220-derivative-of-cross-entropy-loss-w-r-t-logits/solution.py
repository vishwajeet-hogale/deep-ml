import torch
import torch.nn.functional as F

def cross_entropy_derivative(logits: torch.Tensor, target: int) -> torch.Tensor:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs tensor
        target: Index of the true class
        
    Returns:
        Gradient tensor
    """
    # Your code here - can use autograd or the analytical formula
    logits.requires_grad = True
    log_probs = torch.log_softmax(logits, dim=-1)
    one_hot = F.one_hot(torch.tensor(target), num_classes = logits.shape[-1])
    loss = (-1 * one_hot * log_probs[target]).sum()

    loss.backward()
    return logits.grad