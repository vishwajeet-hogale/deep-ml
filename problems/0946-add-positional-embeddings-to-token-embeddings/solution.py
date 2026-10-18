import numpy as np

def add_positional_embeddings(token_embeddings: np.ndarray, pos_embedding_matrix: np.ndarray) -> np.ndarray:
    seq_len, d_model = token_embeddings.shape[-2], token_embeddings.shape[-1]

    if pos_embedding_matrix.shape[1] != d_model:
        raise ValueError("Embedding dims do not match")
    if seq_len > pos_embedding_matrix.shape[0]:
        raise ValueError("Sequence longer than max positions")

    return token_embeddings + pos_embedding_matrix[:seq_len]