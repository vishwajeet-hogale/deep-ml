import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    samples = X[batch_indices, :]        # (B, d)
    y_true  = y[batch_indices]           # (B,)

    y_pred   = samples @ weights + bias  # (B,)
    residual = y_pred - y_true           # (B,)
    error    = np.mean(residual**2)

    grad_w = 2 * samples.T @ residual / len(batch_indices)   # (d,)
    grad_b = 2 * residual.mean()                             # scalar

    weights -= lr * grad_w
    bias    -= lr * grad_b

    return weights.tolist() + [float(bias)]





