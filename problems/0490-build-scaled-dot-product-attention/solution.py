import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 mask: np.ndarray = None) -> tuple:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: (..., seq_len_q, d_k)
        K: (..., seq_len_k, d_k)
        V: (..., seq_len_k, d_v)
        mask: optional binary mask (..., seq_len_q, seq_len_k), 0 = blocked

    Returns:
        (output, attention_weights)
    """
    d_k = Q.shape[-1]
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)

    # numerically stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(scores)
    weights = exp / np.sum(exp, axis=-1, keepdims=True)

    return weights @ V, weights