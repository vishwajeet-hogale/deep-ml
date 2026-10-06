import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    max_vals, _ = t.max(dim=dim, keepdim=True)
    fixed_t = t - max_vals
    exp_fixed_t = torch.exp(fixed_t)

    sum_exp = exp_fixed_t.sum(dim=dim, keepdim=True)
    sm = exp_fixed_t / sum_exp

    return sm


