class SGDOptimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for layer in self.parameters:
            layer.zero_grad()
        
    def step(self):
        for layer in self.parameters:
            assert layer.weight.data.shape == layer.weight.grad.shape
            assert layer.bias.data.shape == layer.bias.grad.shape

            layer.weight.data = layer.weight.data - self.lr * layer.weight.grad
            layer.bias.data = layer.bias.data - self.lr * layer.bias.grad