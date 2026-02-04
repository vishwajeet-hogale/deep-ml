import numpy as np
from typing import Tuple

def compute_qkv(
    X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    return Q, K, V

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # stable softmax
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Scaled dot-product attention (single head).
    Q, K, V: (seq_len, d_k)
    returns: (seq_len, d_k)
    """
    d_k = Q.shape[1]  # scale by feature dim, not seq_len
    scores = (Q @ K.T) / np.sqrt(d_k)          # (seq_len, seq_len)
    weights = softmax(scores, axis=-1)         # (seq_len, seq_len)
    return weights @ V                         # (seq_len, d_k)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndar