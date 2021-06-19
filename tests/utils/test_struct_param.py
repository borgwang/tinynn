import numpy as np

from tinynn.core.layer import Dense
from tinynn.core.net import Net
from tinynn.core.initializer import Uniform
from tinynn.utils.structured_param import StructuredParam
from tinynn.utils.seeder import random_seed

random_seed(0)


def test_structured_param():
    initializer = Uniform(0., 1.)
    net = Net([Dense(10, w_init=initializer, b_init=initializer)])
    net.init_params(input_shape=(1, ))

    params = net.params
    assert isinstance(params, StructuredParam)
    assert isinstance(params.values, np.ndarray)
    assert len(params.values) == 2  # w and b
    assert params.shape[0] == {"w": (1, 10), "b": (10,)}

    # test operations on StructParam
    params = params.clip(0.5, 1.)
    check = (0.5 <= params <= 1.)
    assert isinstance(check, StructuredParam)
    assert check.values[1].all() and check.values[0].all()

    assert((params * 10).values[1] == (10 * params).values[1]).all()
    bigger = params * 10
    assert isinstance(bigger, StructuredParam)
    assert (params.values[1] * 10 == bigger.values[1]).all()

    smaller = params / 10
    assert isinstance(bigger, StructuredParam)
    assert (params.values[1] / 10 == smaller.values[1]).all()

    assert (bigger > smaller).values[1].all()
    assert (smaller < bigger).values[1].all()
    assert ((10 / params).values[1] == (10 / params.values[1])).all()

    power = params ** 2
    assert id(power) != id(params)
    assert np.allclose(power.values[1], params.values[1] ** 2)
    power **= 2
    assert np.allclose(power.values[1], params.values[1] ** 4)

    neg = -params
    assert (neg.values[1] == -params.values[1]).all()

    sum_ = bigger + params
    params2 = sum_ - bigger
    assert np.allclose(params2.values[1], params.values[1])
