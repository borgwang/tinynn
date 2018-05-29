# Author: borgwang <borgwang@126.com>
# Date: 2018-05-05
#
# Filename: layers.py
# Description: Network layers and Activation layers...


import numpy as np

from core.initializer import XavierNormalInit, ZerosInit
from core.math import *


class Layer(object):

    def __init__(self, name):
        self.params = {}
        self.grads = {}
        self.shape = None
        self.name = name

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def initializate_layer(self):
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
        super().__init__('Linear')
        # self.params['w'] = w_init((num_in, num_out))
        # self.params['b'] = b_init((1, num_out))
        self.w_shape = (num_in, num_out)
        self.b_shape = (1, num_out)
        self.w_init = w_init
        self.b_init = b_init
        self.shape = [self.w_shape, self.b_shape]

        self.params = {'w': None, 'b': None}
        self.is_init = False

    def forward(self, inputs):
        if not self.is_init:
            raise ValueError('Parameters unintialized error!')
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad):
        self.grads['w'] = self.inputs.T @ grad
        self.grads['b'] = np.sum(grad, axis=0)
        return grad @ self.params['w'].T

    def initializate(self):
        self.params['w'] = self.w_init(self.w_shape)
        self.params['b'] = self.b_init(self.b_shape)
        self.is_init = True


# ----------
# Non-linear Activation Layers
# ----------

class Activation(Layer):

    def __init__(self, f, f_prime, name):
        super().__init__(name)
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad):
        return self.f_prime(self.inputs) * grad

    def initializate(self):
        return


class Sigmoid(Activation):

    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime, 'Sigmoid')

class Tanh(Activation):

    def __init__(self):
        super().__init__(tanh, tanh_prime, 'Tanh')


class ReLU(Activation):

    def __init__(self):
        super().__init__(relu, relu_prime, 'ReLU')


class LeakyReLU(Activation):

    def __init__(self, negative_slope=0.01):
        super().__init__(leaky_relu, leaky_relu_prime, 'LeakyReLU')
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
        super().__init__('Dropout')
        self._keep_prob = keep_prob
        self.training = True

    def forward(self, inputs):
        self._mask = np.random.binomial(1, self._keep_prob, size=inputs.shape) / self._keep_prob
        return inputs * self._mask

    def backward(self, grad):
        return grad * self._mask

    def initializate(self):
        return
