import torch

def ridge_loss(X: torch.Tensor, w: torch.Tensor, y_true: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Implements the Ridge Regression Loss Function using PyTorch.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        w: Weight vector of shape (n_features,)
        y_true: True target values of shape (n_samples,)
        alpha: Regularization parameter (lambda)
    
    Returns:
        The Ridge loss value as a scalar tensor
    """
    # Your implementation here
    y_true = y_true.reshape(-1, 1)
    w = w.reshape(-1, 1)
    y_pred = (X @ w)
    n = X.shape[0]
    loss = torch.sum((y_true - y_pred) ** 2, dim = 0)/n
    loss += alpha * torch.sum(w*w)
    return loss

