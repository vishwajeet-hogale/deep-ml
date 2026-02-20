import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def attention(Q, K, V):
    """
    Q: (B, QL, D)
    K: (B, KL, D)
    V: (B, KL, D)
    return: (B, QL, D)
    """
    batch_size, query_len, dim = Q.shape
    _, key_len, _ = K.shape

    # 1) scores: (B, QL, KL)
    scores = Q @ K.transpose(0, 2, 1)
    scores = scores / np.sqrt(dim)

    # 2) attention weights (row-wise over keys): (B, QL, KL)
    weights = softmax(scores, axis=-1)

    # 3) weighted sum of values: (B, QL, D)
    out = weights @ V
    return out