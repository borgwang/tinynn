import numpy as np
import tinynn as tn


tn.seeder.random_seed(31)


def test_softmax():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    a = np.exp(x - np.max(x))
    expect = a / a.sum()
    assert np.allclose(tn.math.softmax(x), expect)

    x = np.array([1e10, 1e10])
    expect = [0.5, 0.5]
    assert np.allclose(tn.math.softmax(x), expect)


def test_log_softmax():
    x = np.random.uniform(0, 1, 10)
    assert np.allclose(tn.math.log_softmax(x), np.log(tn.math.softmax(x)))

    x = np.random.uniform(1e10, 1e10, 10)
    assert np.allclose(tn.math.log_softmax(x), np.log(tn.math.softmax(x)))


def test_sigmoid():
    assert tn.math.sigmoid(1e10) == 1.0
    assert tn.math.sigmoid(-1e10) == 0.0
    assert tn.math.sigmoid(0) == 0.5
