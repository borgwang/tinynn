"""test unit for core/layer.py"""

import runtime_path  # isort:skip

import pytest
from core.layer import *
from core.net import Net
from utils.seeder import random_seed

random_seed(0)


@pytest.mark.parametrize("activation_layer, expect_range",
                         [(Sigmoid(), (0, 1)),
                          (Tanh(), (-1, 1)),
                          (ReLU(), (0, np.inf)),
                          (LeakyReLU(), (-np.inf, np.inf)),
                          (Softplus(), (0, np.inf))])
def test_activation(activation_layer, expect_range):
    """Test expected output range of activation layers"""
    input_ = np.random.normal(size=(100, 5))
    net = Net([Dense(1), activation_layer])
    output = net.forward(input_)
    lower_bound, upper_bound = expect_range
    assert np.all((output >= lower_bound) & (output <= upper_bound))


def test_conv_transpose_2d():
    batch_size = 1
    input_ = np.random.randn(batch_size, 7, 7, 1)

    # test forward and backward correctness
    layer = ConvTranspose2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="VALID")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 22, 22, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape

    layer = ConvTranspose2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="SAME")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 21, 21, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape


def test_conv_2d():
    batch_size = 1
    input_ = np.random.randn(batch_size, 16, 16, 1)

    # test forward and backward correctness
    layer = Conv2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="VALID")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 5, 5, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape

    layer = Conv2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="SAME")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 6, 6, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape

