"""Useful math utilities."""

import numpy as np


def softmax(x, t=1.0, axis=-1):
    x_ = x / t
    x_max = np.max(x_, axis=axis, keepdims=True)
    exps = np.exp(x_ - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def log_softmax(x, t=1.0, axis=-1):
    x_ = x / t
    x_max = np.max(x_, axis=axis, keepdims=True)
    exps = np.exp(x_ - x_max)
    exp_sum = np.sum(exps, axis=axis, keepdims=True)
    return x_ - x_max - np.log(exp_sum)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
