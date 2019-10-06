import runtime_path  # isort:skip

import numpy as np

from core.layer import Dense
from core.net import Net
from core.net import StructuredParam


def test_struct_param():
    net = Net([Dense(10, 1)])
    params = net.params
    assert isinstance(params, StructuredParam)
    assert isinstance(params.values, np.ndarray)
    assert len(params.values) == 2  # w and b

    # test operations on StructParam
    bigger = params * 10
    assert isinstance(bigger, StructuredParam)
    assert params.shape == bigger.shape
    assert (params.values[-1] * 10 == bigger.values[-1]).all()




