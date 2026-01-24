import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = None

    def __getitem__(self, idx):
        return Tensor(self.data[idx])

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value


    def __add__(self, other):
        return AddTensor(self, other)

    def __sub__(self, other):
        return SubtractTensor(self, other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return MulTensor(self, other)
        elif isinstance(other, (int, float)):
            return ScaleTensor(self, other)
        else:
            raise NotImplementedError()

    def __floordiv__(self, other):
        raise NotImplementedError()

    def __truediv__(self, other):
        raise NotImplementedError()

    def __mod__(self, other):
        raise NotImplementedError()

    def __divmod__(self, other):
        raise NotImplementedError()

    def __pow__(self, power):
        return PowTensor(self, power)

    def __neg__(self):
        return NegTensor(self)

    def __pos__(self):
        return PosTensor(self)

    def __matmul__(self, other):
        return MatMulTensor(self, other)

    def __str__(self):
        return str(self.data)

    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def mean(self, axis=None):
        return MeanTensor(self, axis=axis)

    def backward(self, grad=None):
        # print("Backward of tensor: ", self.data,self.grad)
        self.grad = grad

    def relu(self):
        return ReluTensor(self)
    
    def tanh(self):
        return TanhTensor(self)

    def abs(self):
        return AbsTensor(self)


class AbsTensor(Tensor):
    def __init__(self, x):
        super().__init__(np.abs(x.data))
        self._x = x

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        x_grad = grad * np.sign(self._x.data)
        assert x_grad.shape == self._x.shape()
        # print("Backward of absolute",self.data,x_grad)
        self._x.backward(x_grad)

class MulTensor(Tensor):
    def __init__(self, x, y):
        assert x.shape() == y.shape()
        super().__init__(x.data * y.data)
        self._x = x
        self._y = y

    def backward(self, grad=None):
        # print("Backward of muliply")
        if grad is None:
           grad = np.ones_like(self.data)

        grad = np.asarray(grad)
        x_grad = grad * self._y.data
        y_grad = grad * self._x.data

        assert x_grad.shape == self._x.shape
        assert y_grad.shape == self._y.shape
        # print("Backward of multiply",self.data,x_grad,y_grad)
        self._x.backward(x_grad)
        self._y.backward(y_grad)


class ReluTensor(Tensor):
    def __init__(self, x):
        super().__init__(np.maximum(x.data, 0))
        self._x = x

    def backward(self, grad=None):
        
        if grad is None:
            grad = np.ones_like(self.data)

        x_grad = grad * (self._x.data > 0).astype(float)
        assert x_grad.shape == self._x.data.shape
        # print("Backward of relu",self.data,x_grad)
        self._x.backward(x_grad)

class TanhTensor(Tensor):
    def __init__(self, x):
        data = np.tanh(x.data)
        super().__init__(data)
        self._x = x

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        x_grad = grad * (1 - self.data ** 2)
        assert x_grad.shape == self._x.data.shape

        self._x.backward(x_grad)



class MatMulTensor(Tensor):
    def __init__(self, x, y):
        super().__init__(x.data @ y.data)
        self._x = x
        self._y = y

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        grad = np.asarray(grad)
        
        x_grad = grad @ self._y.data.T
        y_grad = self._x.data.T @ grad

        assert (
            x_grad.shape == self._x.data.shape
        ), f"x_grad shape mismatch: {x_grad.shape} != {self._x.data.shape}"
        assert (
            y_grad.shape == self._y.data.shape
        ), f"y_grad shape mismatch: {y_grad.shape} != {self._y.data.shape}"
        # print("Backward of matmul",self.data,x_grad,y_grad)
        self._x.backward(x_grad)
        self._y.backward(y_grad)


class PosTensor(Tensor):
    def __init__(self, x):
        super().__init__(x.data)
        self._x = x

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        x_grad = grad
        assert x_grad.shape == self._x.shape
        # print("Backward of positive", self.data, x_grad)
        self._x.backward(x_grad)


class NegTensor(Tensor):
    def __init__(self, x):
        super().__init__(-x.data)
        self._x = x

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        x_grad = -grad
        assert x_grad.shape == self._x.shape
        # print("Backward of negative", self.data, x_grad)
        self._x.backward(x_grad)


class PowTensor(Tensor):
    def __init__(self, x, power):
        super().__init__(np.power(x.data, power))
        self._x = x
        self.power = power

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        assert grad.shape == self.data.shape

        x_grad = self._x.data ** (self.power - 1.0)
        x_grad *= self.power
        x_grad *= grad

        assert x_grad.shape == self._x.data.shape
        # print("Backward of power", self.data, x_grad)
        self._x.backward(x_grad)


class AddTensor(Tensor):
    def __init__(self, x, y):
        assert x.shape() == y.shape()
        super().__init__(x.data + y.data)
        self._x = x
        self._y = y

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        x_grad = grad
        y_grad = grad
        assert x_grad.shape == self._x.data.shape
        assert y_grad.shape == self._y.data.shape
        # print("Backward of add", self.data, x_grad,y_grad)
        self._x.backward(x_grad)
        self._y.backward(y_grad)


class SubtractTensor(Tensor):
    def __init__(self, x, y):
        assert x.shape() == y.shape()
        super().__init__(x.data - y.data)
        self._x = x
        self._y = y

    def backward(self, grad=None):
        # print("Backward of subtract")
        if grad is None:
            grad = np.ones_like(self.data)

        x_grad = grad
        y_grad = grad
        assert x_grad.shape == self._x.data.shape
        assert y_grad.shape == self._y.data.shape
        # print("Backward of subtract", self.data, x_grad,y_grad)
        self._x.backward(grad)
        self._y.backward(-grad)


class ScaleTensor(Tensor):
    def __init__(self, x, scale: float):
        super().__init__(x.data * scale)
        self._x = x
        self._scale = scale

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        assert grad.shape == self.data.shape
        output_grad = grad * self._scale
        assert output_grad.shape == self._x.data.shape
        # print("Backward of scale", self.data, output_grad)
        self._x.backward(output_grad)


class MeanTensor(Tensor):
    def __init__(self, x, axis=None):
        # Initialize the base class with the mean along the specified axis
        super().__init__(np.mean(x.data, axis=axis))
        self._x = x
        self._axis = axis
        if self._axis is None:
            self._axis_size = np.prod(self._x.data.shape)
        else:
            self._axis_size = self._x.data.shape[self._axis]

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        grad = np.asarray(grad)        
        assert grad.shape == self.data.shape
        output_grad = np.broadcast_to(grad/self._axis_size, self._x.data.shape)
        assert output_grad.shape == self._x.data.shape
        # print("Backward of mean", self.data, output_grad)
        self._x.backward(output_grad)
