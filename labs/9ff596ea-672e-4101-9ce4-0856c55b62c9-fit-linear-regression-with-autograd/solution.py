import torch
import numpy as np


def fit_linear_regression(X, y, lr=0.1, steps=500):
    """Fit y ~= X @ w + b with full-batch GD using only autograd.

    Args:
        X: Float tensor (N, D)
        y: Float tensor (N,) or (N, 1)
        lr: learning rate
        steps: number of gradient descent iterations

    Returns:
        w: Float tensor (D,) learned weights (no grad)
        b: Float tensor scalar learned bias (no grad)
    """
    # TODO: ensure y is shape (N,)
    x = torch.tensor(X, dtype= torch.float32)
    N, D = x.shape
    y = torch.tensor(y, dtype=torch.float32)
    if len(y.shape) > 1:
        y = y.squeeze(-1)
    
    # TODO: initialize w (D,) and b with requires_grad=True
    w = torch.randn(D, dtype=torch.float32)
    w.requires_grad = True
    b = torch.randn(1, dtype=torch.float32, requires_grad = True)
    # TODO: for each step:
    #   - predict, compute MSE loss
    #   - loss.backward()
    #   - manual GD update under torch.no_grad()
    #   - zero gradients
    # TODO: return detached w, b
    for _ in range(steps):
        preds = X @ w + b
        loss = ((preds - y)**2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr*w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

    return (w.detach(), b.detach())
