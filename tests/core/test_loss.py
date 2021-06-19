import numpy as np
import pytest

from tinynn.core.loss import Huber, MAE, MSE, \
    SigmoidCrossEntropy, SoftmaxCrossEntropy
from tinynn.utils.math import sigmoid, softmax
from tinynn.utils.seeder import random_seed

random_seed(0)


@pytest.fixture(name="mock_regression")
def fixture_mock_regression():
    preds = np.array([[1., 2.], [3., 6.]])
    trues = np.array([[2., 1.], [6., 3.]])
    return preds, trues


def test_mse(mock_regression):
    preds, trues = mock_regression
    loss = MSE()
    assert loss.loss(preds, trues) == 5.
    assert (loss.grad(preds, trues) == np.array([[-.5, .5], [-1.5, 1.5]])).all()


def test_mae(mock_regression):
    preds, trues = mock_regression
    loss = MAE()
    assert loss.loss(preds, trues) == 4.
    assert (loss.grad(preds, trues) == np.array([[-.5, .5], [-.5, .5]])).all()


def test_huber(mock_regression):
    preds, trues = mock_regression
    delta = 2
    loss = Huber(delta=delta)
    assert loss.loss(preds, trues) == (1 + (3 - .5 * delta) * delta * 2) / 2.
    assert (loss.grad(preds, trues) == np.array([[-.5, .5], [-1., 1.]])).all()


@pytest.fixture(name="mock_binary_classification")
def fixture_mock_binary_classification():
    logits = np.array([[1.], [2.], [-1.]])
    labels = np.array([[0], [1], [0]])
    return logits, labels


@pytest.fixture(name="mock_classification")
def fixture_mock_classification():
    logits = np.array([[1, 2, 3],
                       [2, 1, 3],
                       [3, 1, 2],])
    labels = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0]])
    return logits, labels


def test_sigmoid_cross_entropy(mock_binary_classification):
    logits, labels = mock_binary_classification
    weights = [1, 2]
    loss = SigmoidCrossEntropy(weights=weights)
    p1, p2, p3 = sigmoid(1), sigmoid(2), sigmoid(-1)
    w1, w2 = weights
    expect_loss = -(w1 * np.log(1 - p1) + w2 * np.log(p2) + w1 * np.log(1 - p3))
    expect_loss /= 3.
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


def test_softmax_cross_entropy(mock_classification):
    logits, labels = mock_classification
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
