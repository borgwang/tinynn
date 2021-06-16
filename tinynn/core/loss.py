"""Loss functions"""

import numpy as np
from tinynn.utils.math import log_softmax
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
        batch_size = predicted.shape[0]
        mse_grad = err / batch_size
        mae_grad = np.sign(err) / batch_size
        return (mae_grad * mae_mask + mse_grad * mse_mask) / batch_size


class SoftmaxCrossEntropy(Loss):

    def __init__(self, T=1.0, weight=None):
        """
        L = weight[class] * (-log(exp(x[class]) / sum(exp(x))))
        :paras T: temperature
        :param weight: A 1D tensor [n_classes] assigning weight to each corresponding sample.
        """
        weight = np.asarray(weight) if weight is not None else weight
        self._weight = weight
        self._T = T

    def loss(self, logits, labels):
        nll = -(log_softmax(logits, t=self._T, axis=1) * labels).sum(axis=1)
        if self._weight is not None:
            nll *= self._weight[labels]
        return np.sum(nll) / logits.shape[0]

    def grad(self, logits, labels):
        return (softmax(logits, t=self._T) - labels) / logits.shape[0]


class SigmoidCrossEntropy(Loss):
    """
    logits = a, label = y
    L = -y * log(1 / (1 + exp(-a)) - (1-y) * log(exp(-a) / (1 + exp(-a))
      = -y * a + log(1 + exp(a))

    In order to get stable version, we can further derive
    L = -y * a + log((1 + exp(-a)) / exp(-a))
      = -y * a + log(1 + exp(-a)) + a
    """
    def __init__(self, weight=None):
        weight = np.asarray(weight) if weight is not None else weight
        self._weight = weight

    def loss(self, logits, labels):
        cost = -labels * logits + np.log(1 + np.exp(-logits)) + logits
        return np.sum(cost) / logits.shape[0]

    def grad(self, logits, labels):
        grad = -labels + 1.0 / (1 + np.exp(-logits))
        return grad / logits.shape[0]
