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
