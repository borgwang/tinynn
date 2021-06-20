import numpy as np
import pytest

from tinynn.core.layer import Conv2D, Dense, ReLU, MaxPool2D, Dropout, Flatten
from tinynn.core.loss import SoftmaxCrossEntropy
from tinynn.core.model import Model
from tinynn.core.net import Net
from tinynn.core.optimizer import Adam
from tinynn.utils.dataset import MNIST
from tinynn.utils.metric import accuracy
from tinynn.utils.seeder import random_seed


@pytest.fixture(name="dense_model")
def fixture_dense_model():
    net = Net([
        Dense(128),
        ReLU(),
        Dense(10)
    ])
    return Model(net=net, loss=SoftmaxCrossEntropy(), optimizer=Adam(1e-3))


@pytest.fixture(name="conv_model")
def fixture_conv_model():
    net = Net([
        Conv2D(kernel=[3, 3, 1, 8], padding="VALID"),
        ReLU(),
        MaxPool2D(pool_size=[2, 2]),
        Conv2D(kernel=[3, 3, 8, 16], padding="VALID"),
        ReLU(),
        MaxPool2D(pool_size=[2, 2]),
        Flatten(),
        Dense(10)
    ])
    return Model(net=net, loss=SoftmaxCrossEntropy(), optimizer=Adam(1e-3))


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
    return accuracy(pred_idx, y_idx)["accuracy"]


def test_mnist(dense_model, conv_model, tmpdir):
    random_seed(31)
    mnist = MNIST(tmpdir, one_hot=True)
    X, y = mnist.train_set
    mnist_train(dense_model, X, y, steps=1000, batch_size=128)
    test_X, test_y = mnist.test_set
    acc = mnist_evaluate(dense_model, test_X, test_y)
    assert acc > 0.95

    random_seed(31)
    X, test_X = X.reshape((-1, 28, 28, 1)), test_X.reshape((-1, 28, 28, 1))
    mnist_train(conv_model, X, y, steps=400, batch_size=128)
    acc = mnist_evaluate(conv_model, test_X, test_y)
    assert acc > 0.95
