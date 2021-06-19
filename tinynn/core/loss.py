"""Loss functions"""

import numpy as np
from tinynn.utils.math import log_softmax
from tinynn.utils.math import sigmoid
from tinynn.utils.math import softmax


class Loss:

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, *args, **kwargs):
        raise NotImplementedError


class MSE(Loss):

    def loss(self, predicted, actual):
        return 0.5 * np.sum((predicted - actual) ** 2) / predicted.shape[0]

    def grad(self, predicted, actual):
        return (predicted - actual) / predicted.shape[0]


class MAE(Loss):

    def loss(self, predicted, actual):
        return np.sum(np.abs(predicted - actual)) / predicted.shape[0]

    def grad(self, predicted, actual):
        return np.sign(predicted - actual) / predicted.shape[0]


class Huber(Loss):

    def __init__(self, delta=1.0):
        self._delta = delta

    def loss(self, predicted, actual):
        l1_dist = np.abs(predicted - actual)
        mse_mask = l1_dist < self._delta  # MSE part
        mae_mask = ~mse_mask  # MAE part
        mse = 0.5 * (predicted - actual) ** 2
        mae = self._delta * l1_dist - 0.5 * self._delta ** 2
        return np.sum(mse * mse_mask + mae * mae_mask) / predicted.shape[0]

    def grad(self, predicted, actual):
        err = predicted - actual
        mse_mask = np.abs(err) < self._delta  # MSE part
        mae_mask = ~mse_mask  # MAE part
        mse_grad = err
        mae_grad = np.sign(err) * self._delta
        return (mae_grad * mae_mask + mse_grad * mse_mask) / predicted.shape[0]


class SoftmaxCrossEntropy(Loss):

    def __init__(self, T=1.0, weights=None):
        """
        L = weights[class] * (-log(exp(x[class]) / sum(exp(x))))
        :paras T: temperature
        :param weights: A 1D tensor [n_classes] assigning weight to each corresponding sample.
        """
        self._weights = np.asarray(weights) if weights is not None else weights
        self._T = T

    def loss(self, logits, labels):
        nll = -(log_softmax(logits, t=self._T, axis=1) * labels).sum(axis=1)
        if self._weights is not None:
            nll *= self._weights[np.argmax(labels, axis=1)]
        return np.sum(nll) / logits.shape[0]

    def grad(self, logits, labels):
        grads = softmax(logits, t=self._T) - labels
        if self._weights is not None:
            grads *= self._weights
        return grads / logits.shape[0]


class SigmoidCrossEntropy(Loss):
    """
    let logits = a, label = y, weights[neg] = w1, weights[pos] = w2
    L = - w2 * y * log(1 / (1 + exp(-a)) - w1 * (1-y) * log(exp(-a) / (1 + exp(-a))
      = w1 * a * (1 - y) - (w2 * y - w1 * (y - 1)) * log(sigmoid(a))
    if w1 == w2 == 1:
    L = a * (1 - y) - log(sigmoid(a))

    G = w1 * sigmoid(a) - w2 * y + (w2 - w1) * y * sigmoid(a)
    if w1 == w2 == 1:
    G = sigmoid(a) - y
    """
    def __init__(self, weights=None):
        weights = np.ones(2, dtype=np.float32) if weights is None else weights
        self._weights = np.asarray(weights)

    def loss(self, logits, labels):
        cost = self._weights[0] * logits * (1 - labels) - \
               (self._weights[1] * labels - self._weights[0] * (labels - 1)) * \
               np.log(sigmoid(logits))
        return np.sum(cost) / logits.shape[0]

    def grad(self, logits, labels):
        grads = self._weights[0] * sigmoid(logits) - self._weights[1] * labels + \
                (self._weights[1] - self._weights[0]) * labels * sigmoid(logits)
        return grads / logits.shape[0]
