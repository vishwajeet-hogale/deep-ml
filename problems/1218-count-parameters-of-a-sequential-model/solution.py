import torch.nn as nn

def count_params():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))