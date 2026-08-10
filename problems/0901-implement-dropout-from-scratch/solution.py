import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    if not training:
        return x


    p_map = torch.rand(x.shape)

    p_mask = torch.where(p_map > p, 1.0, 0.0) / (1.0 - p)

    return p_mask*x 
