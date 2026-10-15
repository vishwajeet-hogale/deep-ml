import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    # anchor, positive, negative = anchor.norm(p=2), positive.norm(p=2), negative.norm(p=2)
    dis_ap = ((anchor - positive).norm(p=2))**2
    dis_an = ((anchor - negative).norm(p=2))**2

    loss = (dis_ap - dis_an + torch.tensor(margin, dtype=anchor.dtype)).clamp(min=0.0)
    return loss.mean()

    