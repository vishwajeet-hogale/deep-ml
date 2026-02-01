import torch
import torch.nn.functional as F

def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    # Hint: conv2d expects input of shape (N, C, H, W) and weight of shape (out_channels, in_channels, kH, kW)
    outputs = []
    H, W = input_matrix.shape
    # Pad input
    x = torch.nn.functional.pad(input_matrix, (padding, padding, padding, padding))
    Hp, Wp = x.shape
    kH, kW = kernel.shape
    # Output dimensions
    out_H = (Hp - kH) // stride + 1
    out_W = (Wp - kW) // stride + 1

    out = torch.empty((out_H, out_W), dtype=input_matrix.dtype, device=input_matrix.device)

    for i in range(out_H):
        for j in range(out_W):
            h0 = i * stride
    