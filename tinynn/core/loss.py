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
        m = predicted.shape[0]
        return 0.5 * np.sum((predicted - actual) ** 2) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        return (predicted - actual) / m


class MAE(Loss):

    def loss(self, predicted, actual):
        m = predicted.shape[0]
        return np.sum(np.abs(predicted - actual)) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        return np.sign(predicted - actual) / m


class Huber(Loss):

    def __init__(self, delta=1.0):
        self._delta = delta

    def loss(self, predicted, actual):
        l1_dist = np.abs(predicted - actual)
        mse_mask = l1_dist < self._delta  # MSE part
        mae_mask = ~mse_mask  # MAE part
        mse = 0.5 * (predicted - actual) ** 2
        mae = self._delta * np.abs(predicted - actual) - 0.5 * self._delta ** 2

        m = predicted.shape[0]
        return np.sum(mse * mse_mask + mae * mae_mask) / m

    def grad(self, predicted, actual):
        err = predicted - actual
        mse_mask = np.abs(err) < self._delta  # MSE part
        mae_mask = ~mse_mask  # MAE part
        m = predicted.shape[0]
        mse_grad = err / m
        mae_grad = np.sign(err) / m
        return (mae_grad * mae_mask + mse_grad * mse_mask) / m


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
        m = logits.shape[0]
        nll = -(log_softmax(logits, t=self._T, axis=1) * labels).sum(axis=1)

        if self._weight is not None:
            nll *= self._weight[labels]
        return np.sum(nll) / m

    def grad(self, logits, labels):
        m = logits.shape[0]
        return (softmax(logits, t=self._T) - labels) / m


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
        m = logits.shape[0]
        cost = -labels * logits + np.log(1 + np.exp(-logits)) + logits
        return np.sum(cost) / m

    def grad(self, logits, labels):
        m = logits.shape[0]
        grad = -labels + 1.0 / (1 + np.exp(-logits))
        return grad / m
