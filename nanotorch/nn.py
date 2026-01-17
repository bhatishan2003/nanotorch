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
class RNNLayer:
    """
    Vanilla RNN (many-to-one)
    Matches: nn.RNN(batch_first=True) + last timestep
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # x_t → h_t
        self.Wx = Tensor(np.random.randn(input_size, hidden_size) * 0.01)
        # h_{t-1} → h_t
        self.Wh = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01)
        self.bias = Tensor(np.zeros((1, hidden_size)))

    def __call__(self, x, h0=None):
        return self.forward(x, h0)

    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.shape()

        if h0 is None:
            h_t = Tensor(np.zeros((batch_size, self.hidden_size)))
        else:
            h_t = h0

        ones = Tensor(np.ones((batch_size, 1)))

        for t in range(seq_len):
            x_t = Tensor(x.data[:, t, :])

            bias_expanded = ones @ self.bias
            h_t = (x_t @ self.Wx + h_t @ self.Wh + bias_expanded).relu()

        return h_t

    def zero_grad(self):
        self.Wx.zero_grad()
        self.Wh.zero_grad()
        self.bias.zero_grad()
