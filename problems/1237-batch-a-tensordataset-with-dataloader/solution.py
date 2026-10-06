import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    # 1. Wrap tensors in a TensorDataset
    dataset = TensorDataset(X, y)
    
    # 2. Build the DataLoader with requested settings
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    num_batches = 0
    first_batch_X_shape_tuple = None
    
    # 3. Iterate over all batches
    for i, (batch_X, batch_y) in enumerate(dataloader):
        num_batches += 1
        
        # 4. Capture the first batch's shape as a clean Python tuple
        if i == 0:
            first_batch_X_shape_tuple = tuple(batch_X.shape)
            
    return num_batches, first_batch_X_shape_tuple
