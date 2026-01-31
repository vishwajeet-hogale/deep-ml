import torch

def pca(data, k) -> torch.Tensor:
    X = torch.as_tensor(data, dtype=torch.float32)

    # center
    X = X - X.mean(dim=0, keepdim=True)

    # STANDARDIZE (this makes it correlation-PCA, which your expected output uses)
    std = X.std(dim=0, keepdim=True, unbiased=False)
    X = X / (std + 1e-12)

    # correlation matrix
    cov = (X.T @ X) / X.shape[0]

    # eigendecomposition
    vals, vecs = torch.linalg.eigh(cov)

    # sort descending + take top-k
    idx = torch.argsort(vals, descending=True)
    components = vecs[:, idx][:, :k]  # (D, k)

    # sign fix using mask: if first non-zero entry is negative, flip
    nonzero = components != 0
    first_idx = nonzero.float().argmax(dim=0)  # (k,)
    first_vals = components[first_idx, torch.arange(k)]
    sign = torch.where(first_vals < 0, -1.0, 1.0)  # (k,)
    components = components * sign  # broadcast over rows

    # round to 4 decimals
    return torch.round(components * 10000) / 10000
