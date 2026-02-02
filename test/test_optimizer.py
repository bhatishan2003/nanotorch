import pytest
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.optimizer import SGDOptimizer,SGDMomentumOptimizer, RMSPropOptimizer, AdamOptimizer, AdaGradOptimizer

# ------------------------------
# Fixtures
# ------------------------------
@pytest.fixture
def tensor_param():
    t = Tensor(np.array([1.0, 2.0, 3.0]))
    t.grad = np.array([0.1, 0.1, 0.1])
    return t

@pytest.fixture
def tensor_param_no_grad():
    t = Tensor(np.array([1.0, 2.0, 3.0]))
    t.grad = None
    return t

# ------------------------------
# Helper to test basic optimizer behavior
# ------------------------------
def basic_step_zero_test(OptimizerClass, tensor_param):
    orig_data = tensor_param.data.copy()
    opt = OptimizerClass([tensor_param], lr=0.1)

    # Step should update parameter
    opt.step()
    assert not np.allclose(tensor_param.data, orig_data), "Step did not update parameter"

    # Zero gradients should reset grad
    opt.zero_grad()
    assert np.all((tensor_param.grad == 0)), "Zero grad did not reset gradients"

# ------------------------------
# Tests for SGDOptimizer
# ------------------------------
def test_sgd_step_zero(tensor_param):
    basic_step_zero_test(SGDOptimizer, tensor_param)

def test_sgd_no_grad(tensor_param_no_grad):
    opt = SGDOptimizer([tensor_param_no_grad])
    # Should not raise error
    opt.step()
    opt.zero_grad()
    assert tensor_param_no_grad.grad is None or np.all(tensor_param_no_grad.grad == 0)

# ------------------------------
# Tests for SGDMomentumOptimizer
# ------------------------------
def test_sgd_momentum_step_zero(tensor_param):
    basic_step_zero_test(SGDMomentumOptimizer, tensor_param)

# ------------------------------
# Tests for RMSPropOptimizer
# ------------------------------
def test_rmsprop_step_zero(tensor_param):
    basic_step_zero_test(RMSPropOptimizer, tensor_param)

# ------------------------------
# Tests for AdamOptimizer
# ------------------------------
def test_adam_step_zero(tensor_param):
    basic_step_zero_test(AdamOptimizer, tensor_param)

# ------------------------------
# Tests for AdaGradOptimizer
# ------------------------------
def test_adagrad_step_zero(tensor_param):
    basic_step_zero_test(AdaGradOptimizer, tensor_param)

# ------------------------------
# Test gradient shape mismatch
# ------------------------------
def test_shape_mismatch():
    t = Tensor(np.array([1.0, 2.0]))
    t.grad = np.array([0.1, 0.1, 0.1])  # wrong shape

    for OptimizerClass in [SGDOptimizer, SGDMomentumOptimizer, RMSPropOptimizer, AdamOptimizer, AdaGradOptimizer]:
        opt = OptimizerClass([t])
        with pytest.raises(AssertionError):
            opt.step()
