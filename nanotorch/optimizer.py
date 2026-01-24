from .tensor import Tensor
import numpy as np
class SGDOptimizer:
    def __init__(self, parameters, lr=0.01):
        """
        layers: list of layers (LinearLayer, RNNLayer, etc.)
        lr: learning rate
        """
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """
        Update all parameters using SGD
        """
        for p in self.parameters:
            # Safety checks
            if p.grad is None:
                continue

            assert p.data.shape == p.grad.shape, (
                f"Gradient shape mismatch: "
                f"{p.data.shape} vs {p.grad.shape}"
            )

            # SGD update
            p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Reset gradients to zero
        """
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)