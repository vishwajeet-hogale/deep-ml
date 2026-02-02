import numpy as np

def softmax(matrix):
    matrix = matrix - np.max(matrix, axis=-1, keepdims=True)
    total = np.sum(np.exp(matrix), axis = -1, keepdims=True)
    return np.divide(np.exp(matrix), total)

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(q, k, v):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    
    sd = q @ k.T
    sd = softmax(sd / np.sqrt(q.shape[1]))
    attention_values = sd @ v
    return attention_values

