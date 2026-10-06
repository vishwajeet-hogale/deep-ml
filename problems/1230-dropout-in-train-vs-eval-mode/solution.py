import torch
import torch.nn as nn


def dropout_demo():
    """Demonstrate Dropout behavior in eval vs train mode.

    Returns:
        tuple: (eval_output, train_nonzero_count)
            eval_output: result of Dropout(ones) in eval mode (identity)
            train_nonzero_count: int count of nonzero elements after Dropout in train mode
    """
    # TODO: seed, ones(10), Dropout(0.5), eval output, train nonzero count
    torch.manual_seed(0)
    x = torch.ones(10)
    dp_layer = nn.Dropout(p=0.5)

    dp_layer.eval()
    eval_output = dp_layer(x)

    dp_layer.train()
    out = dp_layer(x)
    return eval_output, int(torch.count_nonzero(out).item())

