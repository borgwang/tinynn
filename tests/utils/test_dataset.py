import numpy as np
import tinynn as tn


def test_mnist(tmpdir):
    ds = tn.dataset.MNIST(tmpdir, one_hot=False)

    assert ds.train_set[0].shape == (50000, 784)
    assert ds.valid_set[0].shape == (10000, 784)
    assert ds.test_set[0].shape == (10000, 784)

    assert ((ds.train_set[0] >= 0) & (ds.train_set[0] <= 1)).all()
    assert ((ds.train_set[1] >= 0) & (ds.train_set[1] <= 9)).all()


def test_fashion_mnist(tmpdir):
    ds = tn.dataset.FashionMNIST(tmpdir, one_hot=False)

    assert ds.train_set[0].shape == (60000, 784)
    assert ds.test_set[0].shape == (10000, 784)

    assert ((ds.train_set[0] >= 0) & (ds.train_set[0] <= 1)).all()
    assert ((ds.train_set[1] >= 0) & (ds.train_set[1] <= 9)).all()


def test_cifar10(tmpdir):
    ds = tn.dataset.Cifar10(tmpdir, one_hot=False, normalize=True)

    assert ds.train_set[0].shape == (50000, 3072)
    assert ds.test_set[0].shape == (10000, 3072)
    assert ((ds.train_set[0] >= -3.0) & (ds.train_set[0] <= 3.0)).all()
    assert ((ds.train_set[1] >= 0) & (ds.train_set[1] <= 9)).all()
    assert np.abs(ds.train_set[0].mean()) < 1e-3


def test_cifar100(tmpdir):
    ds = tn.dataset.Cifar100(tmpdir, one_hot=False, normalize=False)

    assert ds.train_set[0].shape == (50000, 3072)
    assert ds.test_set[0].shape == (10000, 3072)
    assert ((ds.train_set[0] >= 0.0) & (ds.train_set[0] <= 1.0)).all()
    assert ((ds.train_set[1] >= 0) & (ds.train_set[1] <= 99)).all()
