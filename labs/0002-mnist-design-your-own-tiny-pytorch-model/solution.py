import torch
import torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Using very small channel sizes and bias=False before Batch Norm to save parameters
            self.network = nn.Sequential(
                # Step 1: Initial Conv layer down to 12 channels (1*12*9 = 108 parameters)
                nn.Conv2d(1, 12, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.MaxPool2d(2), # Reduces spatial dimension from 28x28 to 14x14
                
                # Step 2: Depthwise Convolution (12*3*3 = 108 parameters)
                nn.Conv2d(12, 12, kernel_size=3, padding=1, groups=12, bias=False),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                
                # Step 3: Pointwise 1x1 Convolution to change depth to 24 channels (12*24*1*1 = 288 parameters)
                nn.Conv2d(12, 24, kernel_size=1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                
                # Step 4: Spatial Reduction via Global Average Pooling to a 1x1 map
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten() # Yields a flat vector of size 24
            )
            
            # Final Classification Head (24 * 10 weights + 10 biases = 250 parameters)
            self.classification_head = nn.Linear(24, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.network(x)
            output = self.classification_head(features)
            return output

    return TinyNet()
