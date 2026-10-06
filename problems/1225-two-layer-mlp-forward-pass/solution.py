import torch
import torch.nn as nn


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    layers = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )

    with torch.no_grad():
        layers[0].weight.copy_(w1)
        layers[0].bias.copy_(b1)
        layers[2].weight.copy_(w2)
        layers[2].bias.copy_(b2)

        out = layers(x)

    return out.item()