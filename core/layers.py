# Author: borgwang <borgwang@126.com>
# Date: 2018-05-05
#
# Filename: layers.py
# Description: Network layers and Activation layers...

import numpy as np

from core.initializer import XavierNormalInit, ZerosInit
from core.math import *


class Layer(object):

    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


# ----------
# Network Layers
# ----------

class Linear(Layer):

    def __init__(self,
                 num_in,
                 num_out,
                 w_init=XavierNormalInit(),
                 b_init=ZerosInit()):
        super().__init__()
        self.params['w'] = w_init((num_in, num_out))
        self.params['b'] = b_init((1, num_out))

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad):
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T


# ----------
# Non-linear Activation Layers
# ----------

class Activation(Layer):

    def __init__(self, f, f_prime):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad):
        return self.f_prime(self.inputs) * grad


class Sigmoid(Activation):

    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


class Tanh(Activation):

    def __init__(self):
        super().__init__(tanh, tanh_prime)


class ReLU(Activation):

    def __init__(self):
        super().__init__(relu, relu_prime)


class LeakyReLU(Activation):

    def __init__(self, negative_slope=0.01):
        super().__init__(leaky_relu, leaky_relu_prime)
        self._negative_slope = negative_slope

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs, self._negative_slope)

    def backward(self, grad):
        return self.f_prime(self.inputs, self._negative_slope) * grad


# ----------
# Function Layers
# ----------

class Dropout(Layer):

    def __init__(self, keep_prob=0.5):
        super().__init__()
        self._keep_prob = keep_prob
        self.training = True

    def forward(self, inputs):
        self._mask = np.random.binomial(1, self._keep_prob, size=inputs.shape) / self._keep_prob
        return inputs * self._mask

    def backward(self, grad):
        return grad * self._mask
