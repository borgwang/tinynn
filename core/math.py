# Author: borgwang <borgwang@126.com>
# Date: 2018-05-20
#
# Filename: math.py
# Description: Math operation library


import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    y = tanh(x)
    return 1 - y ** 2


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    return x > 0


def leaky_relu(x, slope=0.01):
    x[x < 0] *= slope
    return x


def leaky_relu_prime(x, slope=0.01):
    x[x < 0] = slope
    return x
