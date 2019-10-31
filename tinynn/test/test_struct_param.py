import runtime_path  # isort:skip

import numpy as np

from tinynn.core.layer import Dense
from tinynn.core.net import Net
from tinynn.core.net import StructuredParam


def test_struct_param():
    net = Net([Dense(10)])
    net.init_params(input_shape=(1, ))

    params = net.params
    assert isinstance(params, StructuredParam)
    assert isinstance(params.values, np.ndarray)
    assert len(params.values) == 2  # w and b

    # test operations on StructParam
    bigger = params * 10
    assert isinstance(bigger, StructuredParam)
    assert params.shape == bigger.shape
    assert (params.values[-1] * 10 == bigger.values[-1]).all()

    smaller = params / 10
    assert (params.values[-1] / 10 == smaller.values[-1]).all()

    power = params ** 2
    assert (power.values[-1] == params.values[-1] * params.values[-1]).all()

