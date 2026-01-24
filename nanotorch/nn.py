from .tensor import Tensor
import numpy as np
from abc import ABC, abstractmethod

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

class L1Loss:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        return (y_true - y_pred).abs().mean()
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


    def forward(self,x,hidden_state):
        out_1, out_2 = x @ self.weight_x , hidden_state @ self.weight_h 
        for i in range(x.shape()[0]):
            out_1[i] = out_1[i] + self.bias_x
            out_2[i] = out_2[i] + self.bias_h

        
        return (out_1 + out_2).tanh()

from abc import ABC, abstractmethod

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
