import torch
import torch.nn.functional as F
import math 

def overlapping_max_pool2d(x: torch.Tensor, k: int = 3, s: int = 2) -> torch.Tensor:
    """
    Apply overlapping max pooling using PyTorch.
    Must match the ceil mode behavior described in the problem.
    """
    N, C, H, W = x.shape
    out_H = math.ceil(((H - k ) / s) + 1)
    out_W = math.ceil(((W - k ) / s) + 1)
    out = torch.empty(N, C, out_H, out_W)
    for img_idx in range(N):
        for channel in range(C):
            for h in range(0,out_H * s, s):
                min_kh = min(H, h + k)
                for w in range(0, out_W * s, s):
                    min_kw = min(W, w + k)
                    window_2d = x[img_idx, channel, h:min_kh, w:min_kw]
                    out[img_idx, channel, h // s, w // s] = torch.max(window_2d)
    return out