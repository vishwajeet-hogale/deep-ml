import torch

def bn_eval(x, mean, var, gamma, beta, eps=1e-5):
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta