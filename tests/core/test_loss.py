import numpy as np
import pytest

from tinynn.core.loss import *
from tinynn.utils.math import sigmoid
from tinynn.utils.math import softmax
from tinynn.utils.seeder import random_seed

random_seed(0)


@pytest.fixture
def fake_regression():
    preds = np.array([[1., 2.], [3., 6.]])
    targets = np.array([[2., 1.], [6., 3.]])
    return preds, targets


def test_mse(fake_regression):
    preds, targets = fake_regression
    loss = MSE()
    assert loss.loss(preds, targets) == 5.
    assert (loss.grad(preds, targets) == np.array([[-0.5, 0.5], [-1.5, 1.5]])).all()


def test_mae(fake_regression):
    preds, targets = fake_regression
    loss = MAE()
    assert loss.loss(preds, targets) == 4.
    assert (loss.grad(preds, targets) == np.array([[-0.5, 0.5], [-0.5, 0.5]])).all()


def test_huber(fake_regression):
    preds, targets = fake_regression
    delta = 2
    loss = Huber(delta=delta)
    assert loss.loss(preds, targets) == (0.5 * 2 + (3 - 0.5 * delta) * delta * 2) / 2
    assert (loss.grad(preds, targets) == np.array([[-0.5, 0.5], [-1., 1.]])).all()


@pytest.fixture
def fake_binary_classification():
    logits = np.array([[1.], [2.], [-1.]])
    labels = np.array([[0], [1], [0]])
    return logits, labels


@pytest.fixture
def fake_classification():
    logits = np.array([[1, 2, 3],
                       [2, 1, 3],
                       [3, 1, 2],])
    labels = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0]])
    return logits, labels


def test_sigmoid_cross_entropy(fake_binary_classification):
    logits, labels = fake_binary_classification
    weights = [1, 2]
    loss = SigmoidCrossEntropy(weights=weights)
    p1, p2, p3 = sigmoid(1), sigmoid(2), sigmoid(-1)
    w1, w2 = weights
    expect_loss = -(w1 * np.log(1 - p1) + w2 * np.log(p2) + w1 * np.log(1 - p3)) / 3.
    assert np.abs(loss.loss(logits, labels) - expect_loss) < 1e-5

    expect_grads = np.array([[p1], [p2 - 1], [p3]]) / 3.
    expect_grads = []
    for i in range(3):
        if labels[i] == 0:
            grads = w1 * sigmoid(logits[i])
        else:
            grads = -w2 * (1 - sigmoid(logits[i]))
        expect_grads.append(grads / 3.)
    expect_grads = np.array(expect_grads)
    assert np.allclose(loss.grad(logits, labels), expect_grads)


def test_softmax_cross_entropy(fake_classification):
    logits, labels = fake_classification
    weights = [0.1, 0.1, 0.8]
    loss = SoftmaxCrossEntropy(weights=weights)
    expect_loss = 0.
    for i in range(3):
        ci = np.argmax(labels[i])
        expect_loss += (-weights[ci] * np.log(softmax(logits[i])[ci]))
    expect_loss /= 3.
    assert np.abs(loss.loss(logits, labels) - expect_loss) < 1e-5

    expect_grads = []
    for i in range(3):
        ci = np.argmax(labels[i])
        probs = softmax(logits[i])
        for j in range(3):
            if j == ci:
                probs[j] = probs[j] - 1
            probs[j] *= weights[j]
        grads = probs / 3.
        expect_grads.append(grads)
    expect_grads = np.array(expect_grads)
    assert np.allclose(loss.grad(logits, labels), expect_grads)
