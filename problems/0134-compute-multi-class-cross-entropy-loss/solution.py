import torch
import torch.nn.functional as F


def compute_cross_entropy_loss(predicted_probs: torch.Tensor, true_labels: torch.Tensor, epsilon: float = 1e-15) -> float:
    """Compute average cross-entropy loss for multi-class classification.
    
    Args:
        predicted_probs: Tensor of predicted probabilities (batch_size, num_classes)
        true_labels: One-hot encoded true labels (batch_size, num_classes)
        epsilon: Small value for numerical stability
    
    Returns:
        Average cross-entropy loss as a float
    """
    predicted_probs = predicted_probs.clamp(min=epsilon, max = 1 - epsilon)
    preds_idx = torch.argmax(predicted_probs, dim = -1)
    
    preds_one_hot = F.one_hot(preds_idx, num_classes = predicted_probs.shape[-1])

    log_predicted_probs = torch.log(predicted_probs)

    loss = -1 * (true_labels * log_predicted_probs).sum(dim=-1)
    return loss.mean().item()

