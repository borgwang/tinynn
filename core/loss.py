# Author: borgwang <borgwang@126.com>
# Date: 2018-05-05
#
# Filename: loss.py
# Description: Implementation of loss funtions in neural network.

import numpy as np


class Loss(object):

    def loss(self, predicted, actual):
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError


class MSELoss(Loss):

    def loss(self, predicted, actual):
        m = predicted.shape[0]
        return np.sum((predicted - actual) ** 2) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        return 2 * (predicted - actual) / m


class CrossEntropyLoss(Loss):
    '''
    L = weight[class] * (-log(exp(x[class]) / sum(exp(x))))

    weight is a 1D tensor assignning weight to each of the classes.
    '''
    def __init__(self, weight=None, sparse=True):
        self._sparse = sparse
        if weight is not None:
            assert len(weight.shape) == 1
            self._weight = np.asarray(weight)
        else:
            self._weight = None

    def loss(self, predicted, actual):
        m = predicted.shape[0]

        exps = np.exp(predicted - np.max(predicted))
        p = exps / np.sum(exps)
        if self._sparse:
            nll = -np.log(np.sum(p * actual, axis=1))
        else:
            nll = -np.log(p[range(m), actual])

        if self._weight is not None:
            nll *= self._weight[actual]
        return np.sum(nll) / m

    def grad(self, predicted, actual):
        m = predicted.shape[0]
        grad = np.copy(predicted)
        if self._sparse:
            grad -= actual
        else:
            grad[range(m), actual] -= 1.0
        return grad / m
