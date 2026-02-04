from .tensor import Tensor
import numpy as np
from abc import ABC, abstractmethod

class BaseLayer(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        """Collect direct Tensor parameters"""
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                params.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Tensor):
                        params.append(item)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Tensor):
                        params.append(item)
        return params
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()


class LinearLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Tensor(np.random.rand(input_size, output_size))
        self.bias = Tensor(np.random.randn(1, output_size))
    
    def forward(self, x):
        return (x @ self.weight) + (Tensor(np.ones((x.shape()[0],1))) @ self.bias)


class MSELoss:
    def __init__(self):
        pass
    
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)
    
    def forward(self, y_true, y_pred):
        return ((y_true - y_pred)**2).mean()


class MAELoss:
    """
    Mean Absolute Error (MAE)
    Can be used as loss or metric
    """
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        return (y_true - y_pred).abs().mean()

class L1Loss:
    def __init__(self):
        pass
    
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)
    
    def forward(self, y_true, y_pred):
        return (y_true - y_pred).abs().mean()

class EarlyStopping:
    """
    Early stopping based on training loss
    """
    def __init__(self, patience=10, min_delta=0.0):
        """
        patience: epochs to wait after last improvement
        min_delta: minimum loss decrease to be considered improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss):
        """
        Call once per epoch with training loss
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True


class RNNLayer(BaseLayer):
    """
    Vanilla RNN (many-to-one)
    Matches: nn.RNN(batch_first=True) + last timestep
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # x_t → h_t
        self.weight_x = Tensor(np.random.randn(input_size, hidden_size) * 0.01)
        # h_{t-1} → h_t
        self.weight_h = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01)
        self.bias_x = Tensor(np.zeros((hidden_size)))
        self.bias_h = Tensor(np.zeros((hidden_size)))
    
    def forward(self, x, hidden_state):
        out_1, out_2 = x @ self.weight_x, hidden_state @ self.weight_h 
        for i in range(x.shape()[0]):
            out_1[i] = out_1[i] + self.bias_x
            out_2[i] = out_2[i] + self.bias_h
        return (out_1 + out_2).tanh()
    


class Conv2DLayer(BaseLayer):
    """
    2D Convolution Layer (with stride)
    Input:  (B, C_in, H, W)
    Output: (B, C_out, H_out, W_out)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        kh, kw = kernel_size

        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kh, kw) * 0.01
        )
        self.bias = Tensor(np.zeros(out_channels))

    def forward(self, x):
        B, C, H, W = x.shape()
        kh, kw = self.kernel_size
        s = self.stride

        H_out = (H - kh) // s + 1
        W_out = (W - kw) // s + 1

        out = np.zeros((B, self.out_channels, H_out, W_out))

        x_np = x.data
        w_np = self.weight.data
        b_np = self.bias.data

        for b in range(B):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        region = x_np[b, :, h_start:h_start+kh, w_start:w_start+kw]
                        out[b, oc, i, j] = np.sum(region * w_np[oc]) + b_np[oc]

        return Tensor(out)

class BaseModel(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        """Collect parameters from direct child layers"""
        params = []
        for value in self.__dict__.values():
            if isinstance(value, BaseLayer):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, BaseLayer):
                        params.extend(item.parameters())
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, BaseLayer):
                        params.extend(item.parameters())
        return params