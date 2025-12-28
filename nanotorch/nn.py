from .tensor import Tensor
import numpy as np

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Tensor(np.random.rand(input_size, output_size))
        self.bias = Tensor(np.random.randn(1, output_size))
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return (x @ self.weight) + (Tensor(np.ones((x.shape()[0],1))) @ self.bias)
    
    def zero_grad(self):
        self.weight.zero_grad()
        self.bias.zero_grad()

class MSELoss:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        return ((y_true - y_pred)**2).mean()

class L1Loss:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        return (y_true - y_pred).abs().mean()