import torch

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        # Your code here
        self.p = p 
        self.mask = None

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        if not training or self.p == 0:
            self.mask = None
            return x

        keep = 1.0 - self.p
        # 1 where kept, 0 where dropped, then scale kept units by 1/keep
        self.mask = (torch.rand_like(x) >= self.p).to(x.dtype) / keep
        return x * self.mask

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        if self.mask is None:
            return grad

        return grad*self.mask


        