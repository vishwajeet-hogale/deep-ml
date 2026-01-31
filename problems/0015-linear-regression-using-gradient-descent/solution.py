import torch
import torch.nn.functional as F


def criterion(y_pred, y_t) -> torch.Tensor:
    return torch.sum(torch.pow((y_pred - y_t),2), dim = 0) * 0.5* (1/y_pred.shape[0])
def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """
    Perform linear regression using gradient descent with PyTorch autograd.

    Args:
        X: Feature matrix (m, n) - can be tensor or array-like
        y: Target vector (m,) - can be tensor or array-like  
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D tensor of shape (n,)
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True)
    
    # Your code here: use autograd to compute gradients
    for _ in range(iterations):
        y_pred = X_t.matmul(theta)
        loss = criterion(y_pred, y_t)
  