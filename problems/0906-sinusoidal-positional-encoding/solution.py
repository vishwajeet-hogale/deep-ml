import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: return a (seq_len, d_model) tensor of sinusoidal positional encodings
    PE = torch.empty(seq_len, d_model, dtype= torch.float64)


    denominator = torch.arange(0, d_model, 2)
    denominator = torch.pow(10000, denominator / d_model)[None, ...]

    pos = torch.arange(seq_len)
    pos = pos[..., None]
    PE[:, 0::2] = torch.sin(pos / denominator)
    PE[:, 1::2] = torch.cos(pos / denominator)

    return PE
