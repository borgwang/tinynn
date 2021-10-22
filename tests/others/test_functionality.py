import numpy as np
import pytest
import tinynn as tn

tn.seeder.random_seed(31)


@pytest.fixture(name="mock_dataset")
def fixture_mock_dataset():
    X = np.random.normal(size=(100, 64))
    y = np.random.uniform(size=(100, 1))
    return X, y


@pytest.fixture(name="dense")
def fixture_dense_model():
    net = tn.net.Net([tn.layer.Dense(10), tn.layer.Dense(1)])
    loss = tn.loss.MSE()
    opt = tn.optimizer.SGD()
    return tn.model.Model(net, loss, opt)


@pytest.fixture(name="conv")
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


@pytest.fixture(name="rnn")
def fixture_rnn_model():
    net = tn.net.Net([
        tn.layer.RNN(num_hidden=10),
        tn.layer.ReLU(),
        tn.layer.Dense(1)
    ])
    loss = tn.loss.MSE()
    opt = tn.optimizer.SGD()
    return tn.model.Model(net, loss, opt)


@pytest.fixture(name="lstm")
def fixture_lstm_model():
    net = tn.net.Net([
        tn.layer.LSTM(num_hidden=10),
        tn.layer.ReLU(),
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
        assert not np.array_equal(p1, p2)


def test_parameters_change_dense_model(dense, mock_dataset):
    _test_parameter_change(dense, *mock_dataset)


def test_parameter_change_conv_model(conv, mock_dataset):
    X, y = mock_dataset
    X = X.reshape((-1, 8, 8, 1))
    _test_parameter_change(conv, X, y)


def test_parameter_change_rnn_model(rnn, mock_dataset):
    X, y = mock_dataset
    X = X.reshape((-1, 8, 8))
    _test_parameter_change(rnn, X, y)


def test_parameter_change_lstm_model(lstm, mock_dataset):
    X, y = mock_dataset
    X = X.reshape((-1, 8, 8))
    _test_parameter_change(lstm, X, y)


def _test_backprop(model, X, y):
    previous_loss = np.inf
    for _ in range(50):
        pred = model.forward(X)
        loss, grads = model.backward(pred, y)
        model.apply_grads(grads)
        # loss should decrease monotonically
        assert loss < previous_loss
        previous_loss = loss


def test_backprop_dense(dense, mock_dataset):
    _test_backprop(dense, *mock_dataset)


def test_backprop_conv(conv, mock_dataset):
    X, y = mock_dataset
    X = X.reshape((-1, 8, 8, 1))
    _test_backprop(conv, X, y)


def test_backprop_rnn(rnn, mock_dataset):
    X, y = mock_dataset
    X = X.reshape((-1, 8, 8))
    _test_backprop(rnn, X, y)


def test_backprop_lstm(lstm, mock_dataset):
    X, y = mock_dataset
    X = X.reshape((-1, 8, 8))
    _test_backprop(lstm, X, y)
