import numpy as np

def masked_ce_loss(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not mask.any():                      # no response tokens -> avoid mean of empty array
        return 0.0

    logits = logits[mask]                   # (k, vocab_size)  FIX: index rows directly
    targets = targets[mask]                 # (k,)

    # stable log-softmax (same max-shift trick as your earlier function)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    # pick log-prob of the correct token at each position
    nll = -log_probs[np.arange(len(targets)), targets]
    return float(nll.mean())