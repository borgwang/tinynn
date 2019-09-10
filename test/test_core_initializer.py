"""test unit for core/initializer.py"""

import runtime_path  # isort:skip

from core.initializer import *

TEST_SHAPE = (100000, 1)
TOR = 1e-2


def test_get_fans():
    fan_in, fan_out = get_fans(shape=(100, 10))
    assert fan_in == 100 and fan_out == 10

    fan_in, fan_out = get_fans(shape=(64, 5, 5, 128))
    assert fan_in == 5 * 5 * 128
    assert fan_out == 64


def test_normal_init():
    val = NormalInit(mean=0.0, std=1.0).init(TEST_SHAPE)
    assert -TOR <= val.mean() <= TOR
    assert 1.0 - TOR <= val.std() <= 1.0 + TOR


def test_truncated_normal_init():
    val = TruncatedNormalInit(mean=0.0, std=1.0).init(TEST_SHAPE)
    assert -TOR <= val.mean() <= TOR
    assert all(val >= -2.0) and all(val <= 2.0)


def test_uniform_init():
    val = UniformInit(-1.0, 1.0).init(TEST_SHAPE)
    assert all(val >= -1.0) and all(val <= 1.0)


def test_constant_init():
    val = ConstantInit(3.1).init(TEST_SHAPE)
    assert all(val == 3.1)


def test_xavier_uniform_init():
    val = XavierUniformInit().init(TEST_SHAPE)
    bound = np.sqrt(6.0 / np.sum(get_fans(TEST_SHAPE)))
    assert np.all(val >= -bound) and np.all(val <= bound)


def test_xavier_normal_init():
    val = XavierNormalInit().init(TEST_SHAPE)
    std = np.sqrt(2.0 / np.sum(get_fans(TEST_SHAPE)))
    assert std - TOR <= val.std() <= std + TOR


def test_he_uniform_init():
    val = HeUniformInit().init(TEST_SHAPE)
    bound = np.sqrt(6.0 / get_fans(TEST_SHAPE)[0])
    assert np.all(val >= -bound) and np.all(val <= bound)


def test_he_normal_init():
    val = HeNormalInit().init(TEST_SHAPE)
    std = np.sqrt(2.0 / get_fans(TEST_SHAPE)[0])
    assert std - TOR <= val.std() <= std + TOR
