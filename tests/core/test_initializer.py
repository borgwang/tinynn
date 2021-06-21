import numpy as np
import tinynn as tn

tn.seeder.random_seed(31)

TEST_SHAPE = (100000, 1)
TOR = 1e-2


def test_get_fans():
    fan_in, fan_out = tn.initializer.get_fans(shape=(100, 10))
    assert fan_in == 100 and fan_out == 10

    fan_in, fan_out = tn.initializer.get_fans(shape=(64, 5, 5, 128))
    assert fan_in == 5 * 5 * 128
    assert fan_out == 64


def test_normal_init():
    val = tn.initializer.Normal(mean=0.0, std=1.0).init(TEST_SHAPE)
    assert -TOR <= val.mean() <= TOR
    assert 1.0 - TOR <= val.std() <= 1.0 + TOR


def test_truncated_normal_init():
    val = tn.initializer.TruncatedNormal(-2, 2, mean=0.0, std=1.0).init(TEST_SHAPE)
    assert -TOR <= val.mean() <= TOR
    assert all(val >= -2.0) and all(val <= 2.0)


def test_uniform_init():
    val = tn.initializer.Uniform(-1.0, 1.0).init(TEST_SHAPE)
    assert all(val >= -1.0) and all(val <= 1.0)


def test_constant_init():
    val = tn.initializer.Constant(3.1).init(TEST_SHAPE)
    assert all(val == 3.1)


def test_xavier_uniform_init():
    val = tn.initializer.XavierUniform().init(TEST_SHAPE)
    bound = np.sqrt(6.0 / np.sum(tn.initializer.get_fans(TEST_SHAPE)))
    assert np.all(val >= -bound) and np.all(val <= bound)


def test_xavier_normal_init():
    val = tn.initializer.XavierNormal().init(TEST_SHAPE)
    std = np.sqrt(2.0 / np.sum(tn.initializer.get_fans(TEST_SHAPE)))
    assert std - TOR <= val.std() <= std + TOR


def test_he_uniform_init():
    val = tn.initializer.HeUniform().init(TEST_SHAPE)
    bound = np.sqrt(6.0 / tn.initializer.get_fans(TEST_SHAPE)[0])
    assert np.all(val >= -bound) and np.all(val <= bound)


def test_he_normal_init():
    val = tn.initializer.HeNormal().init(TEST_SHAPE)
    std = np.sqrt(2.0 / tn.initializer.get_fans(TEST_SHAPE)[0])
    assert std - TOR <= val.std() <= std + TOR
