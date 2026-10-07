import torch

class MyTransform:
    def __init__(self, factor=0.8):
        self.factor = factor

    def __call__(self, x):
        return (x - 0.5) * self.factor + 0.5