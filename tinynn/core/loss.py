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

    def loss(self, predictions, targets):
        return 0.5 * np.sum((predictions - targets) ** 2) / targets.shape[0]

    def grad(self, predictions, targets):
        return (predictions - targets) / targets.shape[0]


class MAE(Loss):

    def loss(self, predictions, targets):
        return np.sum(np.abs(predictions - targets)) / targets.shape[0]

    def grad(self, predictions, targets):
        return np.sign(predictions - targets) / targets.shape[0]


class Huber(Loss):

    def __init__(self, delta=1.0):
        self._delta = delta

    def loss(self, predictions, targets):
        l1_dist = np.abs(predictions - targets)
        mse_mask = l1_dist < self._delta  # MSE part
        mae_mask = ~mse_mask  # MAE part
        mse = 0.5 * (predictions - targets) ** 2
        mae = self._delta * l1_dist - 0.5 * self._delta ** 2
        return np.sum(mse * mse_mask + mae * mae_mask) / targets.shape[0]

    def grad(self, predictions, targets):
        err = predictions - targets
        mse_mask = np.abs(err) < self._delta  # MSE part
        mae_mask = ~mse_mask  # MAE part
        mse_grad = err
        mae_grad = np.sign(err) * self._delta
        return (mae_grad * mae_mask + mse_grad * mse_mask) / targets.shape[0]


class SoftmaxCrossEntropy(Loss):

    def __init__(self, T=1.0, weights=None):
        self._weights = np.asarray(weights) if weights is not None else weights
        self._T = T

    def loss(self, logits, labels):
        nll = -(log_softmax(logits, t=self._T, axis=1) * labels).sum(axis=1)
        if self._weights is not None:
            nll *= self._weights[np.argmax(labels, axis=1)]
        return np.sum(nll) / labels.shape[0]

    def grad(self, logits, labels):
        grads = softmax(logits, t=self._T) - labels
        if self._weights is not None:
            grads *= self._weights
        return grads / labels.shape[0]


class SigmoidCrossEntropy(Loss):
    """let logits = a, label = y, weights[neg] = w1, weights[pos] = w2
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
        neg_weight, pos_weight = self._weights
        cost = neg_weight * logits * (1 - labels) - \
               (pos_weight * labels - neg_weight * (labels - 1)) * \
               np.log(sigmoid(logits))
        return np.sum(cost) / labels.shape[0]

    def grad(self, logits, labels):
        neg_weight, pos_weight = self._weights
        grads = neg_weight * sigmoid(logits) - pos_weight * labels + \
                (pos_weight - neg_weight) * labels * sigmoid(logits)
        return grads / labels.shape[0]
