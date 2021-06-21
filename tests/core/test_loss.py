import numpy as np
import pytest
import tinynn as tn

tn.seeder.random_seed(31)


@pytest.fixture(name="mock_regression")
def fixture_mock_regression():
    preds = np.array([[1.0, 2.0], [3.0, 6.0]])
    trues = np.array([[2.0, 1.0], [6.0, 3.0]])
    return preds, trues


def test_mse(mock_regression):
    preds, trues = mock_regression
    loss = tn.loss.MSE()
    assert loss.loss(preds, trues) == 5.0
    assert (loss.grad(preds, trues) == np.array([[-.5, .5], [-1.5, 1.5]])).all()


def test_mae(mock_regression):
    preds, trues = mock_regression
    loss = tn.loss.MAE()
    assert loss.loss(preds, trues) == 4.0
    assert (loss.grad(preds, trues) == np.array([[-.5, .5], [-.5, .5]])).all()


def test_huber(mock_regression):
    preds, trues = mock_regression
    delta = 2
    loss = tn.loss.Huber(delta=delta)
    expect_loss = (1 + (3 - 0.5 * delta) * delta * 2) / 2.0
    expect_grads = np.array([[-0.5, 0.5], [-1.0, 1.]])
    assert loss.loss(preds, trues) == expect_loss
    assert (loss.grad(preds, trues) == expect_grads).all()


@pytest.fixture(name="mock_binary_classification")
def fixture_mock_binary_classification():
    logits = np.array([[1.0], [2.0], [-1.0]])
    labels = np.array([[0], [1], [0]])
    return logits, labels


@pytest.fixture(name="mock_classification")
def fixture_mock_classification():
    logits = np.array([[1.0, 2.0, 3.0],
                       [2.0, 1.0, 3.0],
                       [3.0, 1.0, 2.0]])
    labels = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0]])
    return logits, labels


def test_sigmoid_cross_entropy(mock_binary_classification):
    logits, labels = mock_binary_classification
    weights = [1, 2]
    loss = tn.loss.SigmoidCrossEntropy(weights=weights)
    p1, p2, p3 = tn.math.sigmoid(1), tn.math.sigmoid(2), tn.math.sigmoid(-1)
    w1, w2 = weights
    expect_loss = -(w1 * np.log(1 - p1) + w2 * np.log(p2) + w1 * np.log(1 - p3))
    expect_loss /= 3.0
    assert np.abs(loss.loss(logits, labels) - expect_loss) < 1e-5

    expect_grads = np.array([[p1], [p2 - 1], [p3]]) / 3.
    expect_grads = []
    for i in range(3):
        if labels[i] == 0:
            grads = w1 * tn.math.sigmoid(logits[i])
        else:
            grads = -w2 * (1 - tn.math.sigmoid(logits[i]))
        expect_grads.append(grads / 3.0)
    expect_grads = np.array(expect_grads)
    assert np.allclose(loss.grad(logits, labels), expect_grads)


def test_softmax_cross_entropy(mock_classification):
    logits, labels = mock_classification
    weights = [0.1, 0.1, 0.8]
    loss = tn.loss.SoftmaxCrossEntropy(weights=weights)
    expect_loss = 0.
    for i in range(3):
        ci = np.argmax(labels[i])
        expect_loss += (-weights[ci] * np.log(tn.math.softmax(logits[i])[ci]))
    expect_loss /= 3.0
    assert np.abs(loss.loss(logits, labels) - expect_loss) < 1e-5

    expect_grads = []
    for i in range(3):
        ci = np.argmax(labels[i])
        probs = tn.math.softmax(logits[i])
        for j in range(3):
            if j == ci:
                probs[j] = probs[j] - 1
            probs[j] *= weights[j]
        grads = probs / 3.0
        expect_grads.append(grads)
    expect_grads = np.array(expect_grads)
    assert np.allclose(loss.grad(logits, labels), expect_grads)
