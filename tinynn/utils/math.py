"""Useful math utilitiesi."""

import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    exp_sum = np.sum(exps, axis=axis, keepdims=True)
    return x - x_max - np.log(exp_sum)
