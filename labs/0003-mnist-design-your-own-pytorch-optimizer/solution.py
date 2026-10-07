import math
import torch
from torch.optim.optimizer import Optimizer

class MyOptimizer(Optimizer):
    """
    Design your own optimizer!
    - You can base it on SGD, RMSProp, Adam, or create something new.
    - Must subclass torch.optim.Optimizer.
    - Only dense gradients are supported.
    """
    def __init__(self, params, lr=1e-3):
        # You can add your own hyperparameters here
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        # self.beta1 = 

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                # --- TODO: implement your own update rule ---
                # Example: simple SGD update
                # p.add_(grad, alpha=-lr)
                # pass
                p -= lr * grad

        return loss
