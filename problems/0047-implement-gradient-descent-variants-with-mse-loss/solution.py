import torch
import torch.nn.functional as F

def gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    learning_rate: float,
    n_epochs: int,
    batch_size: int = 1,
    method: str = "batch",
) -> torch.Tensor:

    X = X.float()
    y = y.float().view(-1)                 # (m,)
    weights = weights.float().view(-1)     # (n,)

    if not weights.requires_grad:
        weights = weights.clone().detach().requires_grad_(True)

    m = X.shape[0]

    def batch_gd():
        nonlocal weights
        for _ in range(n_epochs):
            outputs = X @ weights
            loss = F.mse_loss(outputs, y)
            loss.backward()

            with torch.no_grad():
                weights -= learning_rate * weights.grad

            weights.grad = None
        return weights.detach().cpu()

    def stochastic_gd():
        nonlocal weights
        for _ in range(n_epochs):
            for i in range(m):
                output = X[i] @ weights
       