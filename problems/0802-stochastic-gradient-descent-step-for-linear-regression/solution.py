import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, lr: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    # Get the num samples 
    n_samples = X.shape[0]

    # Iterate through 1->n_iterations
    for i in range(n_iter):
        # Use i % n_samples to find the sample that needs to be used to calculate the loss 
        idx = i % n_samples
        sample = X[idx, :]
        y_true = y[idx]

        y_pred = weights @ sample
        residual = (y_pred - y_true)
        error = residual**2
        grad_w = 2 * residual * sample       # (d,)
        weights -= lr * grad_w 

    return weights.tolist()





