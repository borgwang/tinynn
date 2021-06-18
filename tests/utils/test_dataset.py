import numpy as np
import pytest
import tempfile

from tinynn.utils.dataset import *


def test_mnist():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = MNIST(tmp_dir, one_hot=False)

    assert ds.train_set[0].shape == (50000, 784)
    assert ds.valid_set[0].shape == (10000, 784)
    assert ds.test_set[0].shape == (10000, 784)

    assert ((0 <= ds.train_set[0]) & (ds.train_set[0] <= 1)).all()
    assert ((0 <= ds.train_set[1]) & (ds.train_set[1] <= 9)).all()


def test_fashion_mnist():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = FashionMNIST(tmp_dir, one_hot=False)

    assert ds.train_set[0].shape == (60000, 784)
    assert ds.test_set[0].shape == (10000, 784)

    assert ((0 <= ds.train_set[0]) & (ds.train_set[0] <= 1)).all()
    assert ((0 <= ds.train_set[1]) & (ds.train_set[1] <= 9)).all()


def test_cifar10():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = Cifar10(tmp_dir, one_hot=False, normalize=True)

    assert ds.train_set[0].shape == (50000, 3072)
    assert ds.test_set[0].shape == (10000, 3072)
    assert ((-3. <= ds.train_set[0]) & (ds.train_set[0] <= 3.)).all()
    assert ((0 <= ds.train_set[1]) & (ds.train_set[1] <= 9)).all()
    assert np.abs(ds.train_set[0].mean()) < 1e-3


def test_cifar100():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = Cifar100(tmp_dir, one_hot=False, normalize=False)

    assert ds.train_set[0].shape == (50000, 3072)
    assert ds.test_set[0].shape == (10000, 3072)
    assert ((0. <= ds.train_set[0]) & (ds.train_set[0] <= 1.)).all()
    assert ((0 <= ds.train_set[1]) & (ds.train_set[1] <= 99)).all()
