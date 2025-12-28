from nanotorch.tensor import Tensor,AddTensor
import numpy as np

def test_add():
    x = Tensor(np.array([1, 2, 3]))
    y = Tensor(np.array([4, 5, 6]))
    z = x + y
    assert isinstance(z, AddTensor)
    assert (z.data == np.array([5,7,9])).all()