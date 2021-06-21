import numpy as np
import pytest
import tinynn as tn

tn.seeder.random_seed(31)


@pytest.fixture(name="mock_dataset")
def fixture_mock_dataset():
    X = np.random.normal(size=(100, 5))
    y = np.random.uniform(size=(100, 1))
    return X, y


@pytest.fixture(name="mock_img_dataset")
def fixture_mock_img_dataset():
    X = np.random.normal(size=(100, 8, 8, 1))
    y = np.random.uniform(size=(100, 1))
    return X, y


@pytest.fixture(name="dense_model")
def fixture_dense_model():
    net = tn.net.Net([tn.layer.Dense(10), tn.layer.Dense(1)])
    loss = tn.loss.MSE()
    opt = tn.optimizer.SGD()
    return tn.model.Model(net, loss, opt)


@pytest.fixture(name="conv_model")
def fixture_conv_model():
    net = tn.net.Net([
        tn.layer.Conv2D(kernel=[3, 3, 1, 2]),
        tn.layer.MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        tn.layer.Conv2D(kernel=[3, 3, 2, 4]),
        tn.layer.MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        tn.layer.Flatten(),
        tn.layer.Dense(1)
    ])
    loss = tn.loss.MSE()
    opt = tn.optimizer.SGD()
    return tn.model.Model(net, loss, opt)


def _test_parameter_change(model, X, y):
    pred = model.forward(X)
    loss, grads = model.backward(pred, y)
    # make sure the parameters does change after apply gradients
    params_before = model.net.params.values
    model.apply_grads(grads)
    params_after = model.net.params.values
    for p1, p2 in zip(params_before, params_after):
        assert np.all(p1 != p2)


def test_parameters_change_dense_model(dense_model, mock_dataset):
    _test_parameter_change(dense_model, *mock_dataset)


def test_parameter_change_conv_model(conv_model, mock_img_dataset):
    _test_parameter_change(conv_model, *mock_img_dataset)


def _test_backprop(model, X, y):
    previous_loss = np.inf
    for _ in range(50):
        pred = model.forward(X)
        loss, grads = model.backward(pred, y)
        model.apply_grads(grads)
        # loss should decrease monotonically
        assert loss < previous_loss
        previous_loss = loss


def test_backprop_dense(dense_model, mock_dataset):
    _test_backprop(dense_model, *mock_dataset)


def test_backprop_conv(conv_model, mock_img_dataset):
    _test_backprop(conv_model, *mock_img_dataset)
