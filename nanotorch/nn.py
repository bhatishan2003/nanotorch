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
        self.weight_x = Tensor(np.random.randn(input_size, hidden_size) * 0.01)
        # h_{t-1} → h_t
        self.weight_h = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01)
        self.bias_x = Tensor(np.zeros((hidden_size)))
        self.bias_h = Tensor(np.zeros((hidden_size)))


    def __call__(self, x, hidden_state):
        return self.forward(x, hidden_state)

    def forward(self,x,hidden_state):
        out_1, out_2 = x @ self.weight_x , hidden_state @ self.weight_h 
        for i in range(x.shape()[0]):
            out_1[i] = out_1[i] + self.bias_x
            out_2[i] = out_2[i] + self.bias_h

        
        return (out_1 + out_2).tanh()


    def zero_grad(self):
        self.weight_x.zero_grad()
        self.weight_h.zero_grad()
        self.bias_x.zero_grad()
        self.bias_h.zero_grad()

