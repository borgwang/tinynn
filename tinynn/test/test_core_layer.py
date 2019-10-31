"""test unit for core/layer.py"""

import runtime_path  # isort:skip

import pytest
from tinynn.core.layer import *
from tinynn.core.net import Net
from tinynn.utils.seeder import random_seed

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


def test_max_pool_2d():
    batch_size = 1
    channel = 2
    input_ = np.random.randn(batch_size, 4, 4, channel)

    layer = MaxPool2D(pool_size=[2, 2], stride=[2, 2])
    output = layer.forward(input_)
    assert output.shape == (batch_size, 2, 2, channel)

    layer = MaxPool2D(pool_size=[4, 4], stride=[2, 2])
    output = layer.forward(input_)
    answer = np.max(np.reshape(input_, (batch_size, -1, 2)), axis=1)
    assert (output.ravel() == answer.ravel()).all()


def test_reshape():
    batch_size = 1
    input_ = np.random.randn(batch_size, 2, 3, 4, 5)
    target_shape = (5, 4, 3, 2)
    layer = Reshape(*target_shape)
    output = layer.forward(input_)
    assert output.shape[1:] == target_shape
