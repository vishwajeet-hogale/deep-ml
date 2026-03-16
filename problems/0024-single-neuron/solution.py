import torch
import torch.nn.functional as F
from typing import List, Tuple

def single_neuron_model(
    features: List[List[float]],
    labels: List[float],
    weights: List[float],
    bias: float
) -> Tuple[List[float], float]:
    """
    Compute output probabilities and MSE for a single neuron.
    Uses built-in sigmoid and MSE loss.
    """
    # Your code here
	X = torch.tensor(features, dtype = torch.float32)
	W = torch.tensor(weights, dtype = torch.float32)
	y = torch.tensor(labels, dtype = torch.float32)
	B = torch.tensor(bias, dtype = torch.float32)

	y_pred = torch.sigmoid(W @ X.T + B)

	mse = torch.mean(torch.pow(y_pred - y , 2))

	return y_pred.tolist(), mse.item()
