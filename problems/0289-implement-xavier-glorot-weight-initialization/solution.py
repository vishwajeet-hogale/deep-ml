import math
import torch
import torch.nn as nn

def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> torch.Tensor:
    if len(shape) != 2:
        raise ValueError("shape must be (fan_in, fan_out)")
    fan_in, fan_out = shape

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    if mode == 'uniform':
        a = math.sqrt(6.0 / (fan_in + fan_out))
        return torch.empty(shape).uniform_(-a, a, generator=gen)
    elif mode == 'normal':
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return torch.empty(shape).normal_(0.0, std, generator=gen)
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")