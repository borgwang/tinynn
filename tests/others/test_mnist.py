import numpy as np
import pytest
import tinynn as tn


@pytest.fixture(name="dense_model")
def fixture_dense_model():
    net = tn.net.Net([
        tn.layer.Dense(128),
        tn.layer.ReLU(),
        tn.layer.Dense(10)
    ])
    loss = tn.loss.SoftmaxCrossEntropy()
    optimizer = tn.optimizer.Adam(1e-3)
    return tn.model.Model(net=net, loss=loss, optimizer=optimizer)


@pytest.fixture(name="conv_model")
def fixture_conv_model():
    net = tn.net.Net([
        tn.layer.Conv2D(kernel=[3, 3, 1, 8], padding="VALID"),
        tn.layer.ReLU(),
        tn.layer.MaxPool2D(pool_size=[2, 2]),
        tn.layer.Conv2D(kernel=[3, 3, 8, 16], padding="VALID"),
        tn.layer.ReLU(),
        tn.layer.MaxPool2D(pool_size=[2, 2]),
        tn.layer.Flatten(),
        tn.layer.Dense(10)
    ])
    loss = tn.loss.SoftmaxCrossEntropy()
    optimizer = tn.optimizer.Adam(1e-3)
    return tn.model.Model(net=net, loss=loss, optimizer=optimizer)


def mnist_train(model, X, y, steps, batch_size):
    for _ in range(steps):
        indices = np.random.choice(np.arange(len(X)), size=batch_size,
                                   replace=False)
        pred = model.forward(X[indices])
        loss, grads = model.backward(pred, y[indices])
        model.apply_grads(grads)


def mnist_evaluate(model, X, y):
    pred = model.forward(X)
    pred_idx = np.argmax(pred, axis=1)
    y_idx = np.argmax(y, axis=1)
    accuracy, _ = tn.metric.accuracy(pred_idx, y_idx)
    return accuracy


def test_mnist(dense_model, conv_model, tmpdir):
    tn.seeder.random_seed(31)
    mnist = tn.dataset.MNIST(tmpdir, one_hot=True)
    X, y = mnist.train_set
    mnist_train(dense_model, X, y, steps=1000, batch_size=128)
    test_X, test_y = mnist.test_set
    acc = mnist_evaluate(dense_model, test_X, test_y)
    assert acc > 0.95

    tn.seeder.random_seed(31)
    X, test_X = X.reshape((-1, 28, 28, 1)), test_X.reshape((-1, 28, 28, 1))
    mnist_train(conv_model, X, y, steps=400, batch_size=128)
    acc = mnist_evaluate(conv_model, test_X, test_y)
    assert acc > 0.95
