import torch
import torch.nn.functional as F

def avg_pool_2d(input_matrix: torch.Tensor, pool_size: int) -> torch.Tensor:
    """
    Perform 2D average pooling on the input matrix.
    
    Args:
        input_matrix: 2D input tensor of shape (H, W)
        pool_size: Size of the square pooling window
        
    Returns:
        2D tensor after average pooling of shape (H//pool_size, W//pool_size)
    """
    input_matrix = input_matrix.to(torch.float32)
    H, W = input_matrix.shape
    oH, oW = (H - pool_size) // pool_size + 1, (W - pool_size) // pool_size + 1
    patches = F.unfold(
        input_matrix[None, None, ...], 
        kernel_size=(pool_size, pool_size), 
        stride=(pool_size, pool_size)
    ) # [1, pool_size*pool_size, T]
    patches = patches.transpose(-1,-2)
    patches = patches.mean(dim=-1).reshape(1,oH, oW).squeeze(0)
    


    print("Patches shape:", patches.shape)
    return patches