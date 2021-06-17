"""test unit for utils/math.py"""

import runtime_path  # isort:skip

import numpy as np

from tinynn.utils.math import *
from tinynn.utils.seeder import random_seed

random_seed(0)


def test_softmax():
    x = np.array([1., 2., 3., 4])
    a = np.exp(x - np.max(x))
    expect = a / a.sum()
    assert np.allclose(softmax(x), expect)

    x = np.array([1e10, 1e10])
    expect = [0.5, 0.5]
    assert np.allclose(softmax(x), expect)


def test_log_softmax():
    x = np.random.uniform(0, 1, 10)
    assert np.allclose(log_softmax(x), np.log(softmax(x)))

    x = np.random.uniform(1e10, 1e10, 10)
    assert np.allclose(log_softmax(x), np.log(softmax(x)))


def test_sigmoid():
    assert sigmoid(1e10) == 1.
    assert sigmoid(-1e10) == 0.
    assert sigmoid(0) == 0.5
