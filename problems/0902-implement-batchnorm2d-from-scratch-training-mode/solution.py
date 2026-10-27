import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d
    B, N, H, W = x.shape
    meu = x.mean(dim=(0,2,3), keepdim = True) 
    
    var = x.var(dim=(0,2,3), keepdim=True, unbiased = False)

    return gamma[None, :, None, None] * (x - meu) / torch.sqrt(var + eps) + beta[None, :, None, None]



