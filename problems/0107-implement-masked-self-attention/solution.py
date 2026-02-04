import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    prod = (Q @ K.T) / Q.shape[1] ** (0.5)
    mask = torch.ones_like(prod, dtype = torch.bool).triu(1)
    prod = prod.masked_fill_(mask, -1* torch.inf)
    attention = torch.softmax(prod, dim= -1) @ V
    return attention