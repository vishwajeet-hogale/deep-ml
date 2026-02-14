import torch
from typing import Optional, Union

def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, "np.ndarray"],
    Y: Optional[Union[torch.Tensor, list, "np.ndarray"]] = None
) -> torch.Tensor:
    X = torch.as_tensor(X).to(torch.float32)
    N = X.shape[0]           # samples
    n = X.shape[1]           # features

    mean_x = X.mean(dim=0)   # (n,)
    std_x  = X.std(dim=0)    # (n,)  (unbiased=True by default)

    if Y is None:
        res = torch.zeros(n, n, dtype=torch.float32)
        for i in range(n):
            for j in range(n):
                cov_ij = (1 / (N - 1)) * torch.sum((X[:, i] - mean_x[i]) * (X[:, j] - mean_x[j]))
                res[i, j] = cov_ij / (std_x[i] * std_x[j])
        return res

    Y = torch.as_tensor(Y).to(torch.float32)
    m = Y.shape[1]
    mean_y = Y.mean(dim=0)   # (m,)
    std_y  = Y.std(dim=0)    # (m,)

    res = torch.zeros(n, m, dtype=torch.float32)
    for i in range(n):
        for j in range(m):
            cov_ij