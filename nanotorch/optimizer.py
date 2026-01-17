class SGDOptimizer:
    def __init__(self, layers, lr=0.01):
        """
        layers: list of layers (LinearLayer, RNNLayer, etc.)
        lr: learning rate
        """
        self.layers = layers
        self.lr = lr

    def step(self):
        """
        Update all parameters using SGD
        """
        for layer in self.layers:

            # Generic parameter access (PyTorch-style)
            if hasattr(layer, "parameters"):
                params = layer.parameters()
            else:
                # Fallback for very simple layers
                params = []
                if hasattr(layer, "weight"):
                    params.append(layer.weight)
                if hasattr(layer, "bias"):
                    params.append(layer.bias)

            for p in params:
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
        for layer in self.layers:

            if hasattr(layer, "parameters"):
                params = layer.parameters()
            else:
                params = []
                if hasattr(layer, "weight"):
                    params.append(layer.weight)
                if hasattr(layer, "bias"):
                    params.append(layer.bias)

            for p in params:
                p.grad = 0
