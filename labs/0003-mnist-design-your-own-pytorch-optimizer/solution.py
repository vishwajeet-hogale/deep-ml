import math
import torch
from torch.optim.optimizer import Optimizer

class MyOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.eps = 1e-7
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['m'] = torch.zeros_like(p)
                self.state[p]['v'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.t += 1

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                self.state[p]['m'] = self.beta1 * self.state[p]['m'] + (1 - self.beta1) * grad
                self.state[p]['v'] = self.beta2 * self.state[p]['v'] + (1 - self.beta2) * (grad**2)
                m_hat = self.state[p]['m'] / (1 - (self.beta1 ** self.t))
                v_hat = self.state[p]['v'] / (1 - (self.beta2 ** self.t))
                p -= lr * m_hat / (torch.sqrt(v_hat) + self.eps)

        return loss