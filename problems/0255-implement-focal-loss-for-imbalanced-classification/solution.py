import torch
import torch.nn.functional as F

def focal_loss(y_true: torch.Tensor, y_pred: torch.Tensor, gamma: float = 2.0, alpha: torch.Tensor = None) -> float:
    """
    Compute Focal Loss for multi-class classification.
    
    Args:
        y_true: Ground truth labels as class indices (1D tensor)
        y_pred: Predicted probabilities (2D tensor, shape: [n_samples, n_classes])
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights (optional, 1D tensor of length n_classes)
    
    Returns:
        float: Average focal loss
    """
    # 1. Clip predictions to avoid log(0) and log(1) numerical issues
    probs = torch.clamp(y_pred, min=1e-8, max=1.0 - 1e-8)
    
    # 2. Extract the probability (p_t) corresponding to the TRUE class
    # y_true needs to be unsqueezed to shape [B, 1] for 2D gathering
    p_t = probs.gather(dim=-1, index=y_true.unsqueeze(-1)).squeeze(-1)
    
    # 3. Compute log(p_t) directly from the clipped probabilities
    log_p_t = torch.log(p_t)
    
    # 4. Calculate the core Focal Loss step for each sample
    loss = -1 * ((1 - p_t) ** gamma) * log_p_t
    
    # 5. Apply class-specific alpha weights if provided
    if alpha is not None:
        alpha_t = alpha.gather(dim=-1, index=y_true)
        loss = alpha_t * loss
        
    return float(loss.mean().item())