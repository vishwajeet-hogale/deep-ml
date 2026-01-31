import torch

def batch_normalization(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""
    if x.dim() != 4:
        raise ValueError(f"Expected x to be 4D (B, C, H, W), got {tuple(x.shape)}")

    B, C, H, W = x.shape

    # Allow gamma/beta as (C,) or (1, C, 1, 1)
    if gamma.dim() == 1:
        gamma = gamma.view(1, C, 1, 1)
    if beta.dim() == 1:
        beta = beta.view(1, C, 1, 1)

    # Per-channel mean/var over (B, H, W)
    mean = x.mean(dim=(0, 2, 3), keepdim=True)                 # (1, C, 1, 1)
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)   # (1, C, 1, 1)

    x_hat = (x - mean) / torch.sqrt(var + epsilon)
    out = gamma * x_hat + beta
    return out