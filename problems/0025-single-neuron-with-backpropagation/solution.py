import torch
from typing import List, Tuple, Union


def train_neuron(
    features: Union[List[List[float]], torch.Tensor],
    labels:   Union[List[float],      torch.Tensor],
    weights: Union[List[float], torch.Tensor],
    bias: float,
    lr: float,
    epochs: int
) -> Tuple[List[float], float, List[float]]:

    X = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)

    W = torch.as_tensor(weights, dtype=torch.float32).clone().detach().requires_grad_(True)
    B = torch.tensor(bias, dtype=torch.float32, requires_grad=True)

    mse_list = []

    for _ in range(epochs):

        y_pred = torch.sigmoid(X @ W + B)

        mse = torch.mean((y_pred - y) ** 2)

        mse.backward()

        with torch.no_grad():
            W -= lr * W.grad
            B -= lr * B.grad

        W.grad.zero_()
        B.grad.zero_()

        mse_list.append(round(mse.item(), 4))

    W_out = [round(w, 4) for w in W.detach().tolist()]
    B_out = round(B.item(), 4)

    return W_out, B_out, mse_list