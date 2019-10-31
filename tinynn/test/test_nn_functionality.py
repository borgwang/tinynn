"""Test the functionality of tinynn. Inspired by https://github.com/Thenerdstation/mltest"""

import runtime_path  # isort:skip

import numpy as np

import pytest
from tinynn.core.layer import Conv2D
from tinynn.core.layer import Dense
from tinynn.core.layer import Flatten
from tinynn.core.layer import MaxPool2D
from tinynn.core.loss import MSE
from tinynn.core.model import Model
from tinynn.core.net import Net
from tinynn.core.optimizer import SGD
from tinynn.utils.seeder import random_seed

random_seed(0)


@pytest.fixture
def fake_dataset():
    X = np.random.normal(size=(100, 5))
    y = np.random.uniform(size=(100, 1))
    return X, y


@pytest.fixture
def img_dataset():
    X = np.random.normal(size=(100, 8, 8, 1))
    y = np.random.uniform(size=(100, 1))
    return X, y


@pytest.fixture
def fc_model():
    net = Net([Dense(10), Dense(1)])
    loss = MSE()
    opt = SGD()
    return Model(net, loss, opt)


@pytest.fixture
def cnn_model():
    net = Net([
        Conv2D(kernel=[3, 3, 1, 2]),
        MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        Conv2D(kernel=[3, 3, 2, 4]),
        MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        Flatten(),
        Dense(1)
    ])
    return Model(net, loss=MSE(), optimizer=SGD())


def test_parameters_change(fake_dataset):
    # make sure the parameters does change after apply gradients

    # fake dataset
    X, y = fake_dataset
    # simple model
    net = Net([Dense(10), Dense(1)])
    loss = MSE()
    opt = SGD(lr=1.0)
    model = Model(net, loss, opt)

    # forward and backward
    pred = model.forward(X)
    loss, grads = model.backward(pred, y)

    # parameters change test
    params_before = model.net.params.values
    model.apply_grads(grads)
    params_after = model.net.params.values
    for p1, p2 in zip(params_before, params_after):
        assert np.all(p1 != p2)


def test_backprop_dense(fc_model, fake_dataset):
    # train on a single data point
    X, y = fake_dataset

    previous_loss = np.inf
    for step in range(50):
        pred = fc_model.forward(X)
        loss, grads = fc_model.backward(pred, y)
        fc_model.apply_grads(grads)
        # loss should decrease monotonically
        assert loss < previous_loss
        previous_loss = loss


def test_backprop_cnn(cnn_model, img_dataset):
    # train on a single data point
    X, y = img_dataset

    previous_loss = np.inf
    for step in range(50):
        pred = cnn_model.forward(X)
        loss, grads = cnn_model.backward(pred, y)
        cnn_model.apply_grads(grads)
        # loss should decrease monotonically
        assert loss < previous_loss
        previous_loss = loss
