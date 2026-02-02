from .tensor import Tensor
import numpy as np

# -------------------------------------------------
# SGD Optimizer (Vanilla)
# -------------------------------------------------
class SGDOptimizer:
    def __init__(self, parameters, lr=0.01):
        """
        parameters: list of Tensor parameters
        lr: learning rate
        """
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """
        Update all parameters using vanilla SGD
        """
        for p in self.parameters:
            if p.grad is None:
                continue

            assert p.data.shape == p.grad.shape, (
                f"Gradient shape mismatch: "
                f"{p.data.shape} vs {p.grad.shape}"
            )

            # SGD update rule
            p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Reset gradients to zero
        """
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


# -------------------------------------------------
# SGD with Momentum Optimizer
# -------------------------------------------------
class SGDMomentumOptimizer:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        """
        parameters: list of Tensor parameters
        lr: learning rate
        momentum: momentum coefficient (default 0.9)
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

        # Initialize velocity for each parameter
        self.velocities = {
            id(p): np.zeros_like(p.data) for p in self.parameters
        }

    def step(self):
        """
        Update all parameters using SGD with Momentum
        """
        for p in self.parameters:
            if p.grad is None:
                continue

            assert p.data.shape == p.grad.shape, (
                f"Gradient shape mismatch: "
                f"{p.data.shape} vs {p.grad.shape}"
            )

            v = self.velocities[id(p)]

            # Momentum update
            v[:] = self.momentum * v - self.lr * p.grad

            # Apply update
            p.data += v

    def zero_grad(self):
        """
        Reset gradients to zero
        """
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

class RMSPropOptimizer:
    def __init__(self, parameters, lr=0.001, beta=0.9, eps=1e-8):
        """
        parameters: list of Tensor parameters
        lr: learning rate
        beta: decay rate for squared gradients
        eps: numerical stability term
        """
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self.eps = eps

        # Running average of squared gradients
        self.sq_grads = {
            id(p): np.zeros_like(p.data) for p in self.parameters
        }

    def step(self):
        """
        Update parameters using RMSProp
        """
        for p in self.parameters:
            if p.grad is None:
                continue

            assert p.data.shape == p.grad.shape, (
                f"Gradient shape mismatch: "
                f"{p.data.shape} vs {p.grad.shape}"
            )

            s = self.sq_grads[id(p)]

            # Update running average of squared gradients
            s[:] = self.beta * s + (1.0 - self.beta) * (p.grad ** 2)

            # RMSProp update
            p.data -= self.lr * p.grad / (np.sqrt(s) + self.eps)

    def zero_grad(self):
        """
        Reset gradients to zero
        """
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


# -------------------------------------------------
# Adam Optimizer
# -------------------------------------------------
class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        parameters: list of Tensor parameters
        lr: learning rate
        beta1: decay rate for first moment (mean of gradients)
        beta2: decay rate for second moment (mean of squared gradients)
        eps: numerical stability term
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Time step
        self.t = 0

        # First and second moment estimates
        self.m = {id(p): np.zeros_like(p.data) for p in self.parameters}
        self.v = {id(p): np.zeros_like(p.data) for p in self.parameters}

    def step(self):
        """
        Update parameters using Adam
        """
        self.t += 1

        for p in self.parameters:
            if p.grad is None:
                continue

            assert p.data.shape == p.grad.shape, (
                f"Gradient shape mismatch: "
                f"{p.data.shape} vs {p.grad.shape}"
            )

            m = self.m[id(p)]
            v = self.v[id(p)]

            # Update biased first moment estimate
            m[:] = self.beta1 * m + (1.0 - self.beta1) * p.grad

            # Update biased second moment estimate
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (p.grad ** 2)

            # Bias correction
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            # Adam update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """
        Reset gradients to zero
        """
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

# -------------------------------------------------
# AdaGrad Optimizer
# -------------------------------------------------
class AdaGradOptimizer:
    def __init__(self, parameters, lr=0.01, eps=1e-8):
        """
        parameters: list of Tensor parameters
        lr: learning rate
        eps: numerical stability term
        """
        self.parameters = parameters
        self.lr = lr
        self.eps = eps

        # Accumulated squared gradients
        self.accum_grads = {
            id(p): np.zeros_like(p.data) for p in self.parameters
        }

    def step(self):
        """
        Update parameters using AdaGrad
        """
        for p in self.parameters:
            if p.grad is None:
                continue

            assert p.data.shape == p.grad.shape, (
                f"Gradient shape mismatch: "
                f"{p.data.shape} vs {p.grad.shape}"
            )

            g2 = self.accum_grads[id(p)]

            # Accumulate squared gradients
            g2[:] += p.grad ** 2

            # AdaGrad update
            p.data -= self.lr * p.grad / (np.sqrt(g2) + self.eps)

    def zero_grad(self):
        """
        Reset gradients to zero
        """
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
