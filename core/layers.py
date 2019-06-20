# Author: borgwang <borgwang@126.com>
# Date: 2018-05-05
#
# Filename: layers.py
# Description: Network layers and Activation layers...


from core.initializer import XavierNormalInit
from core.initializer import ZerosInit
from core.math import *


class Layer(object):

    def __init__(self, name):
        self.params, self.grads = {}, {}
        self.shape = None
        self.name = name
        self.is_training = True

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def set_phase(self, phase):
        self.is_training = True if phase == 'TRAIN' else False

    @property
    def out_dim(self):
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

    def initialize(self):
        self.params['w'] = self.w_init(self.w_shape)
        self.params['b'] = self.b_init(self.b_shape)
        self.is_init = True


# class Conv2D(Layer):
#
#     def __init__(self,
#                  in_dim,
#                  channels,
#                  kernel_size,
#                  stride=1,
#                  padding='SAME',
#                  w_init=XavierNormalInit(),
#                  b_init=ZerosInit()):
#         super().__init__('Conv2D')
#         self.h_in, self.w_in, self.d_in = in_dim
#         # https://www.tensorflow.org/api_guides/python/nn#convolution
#         if padding == 'SAME':
#             self.h_out = np.ceil(self.h_in / stride)
#             self.w_out = np.ceil(self.w_in / stride)
#             h_pad_needed = int((self.h_out - 1) * stride + kernel_size - self.h_in)
#             pad_top = int(h_pad_needed / 2)
#             pad_bottom = h_pad_needed - pad_top
#             w_pad_needed = int((self.w_out - 1) * stride + kernel_size - self.h_in)
#             pad_left = int(w_pad_needed / 2)
#             pad_right = w_pad_needed - pad_left
#             self.pad_needed = (pad_top, pad_bottom, pad_left, pad_right)
#         elif padding == 'VALID':
#             self.h_out = np.ceil((self.h_in - kernel_size + 1) / stride)
#             self.w_out = np.ceil((self.w_in - kernel_size + 1) / stride)
#             self.pad_needed = (0, 0, 0, 0)
#         else:
#             raise ValueError('Invalid padding mode.')
#
#         self.h_f, self.w_f, self.n_f = kernel_size, kernel_size, channels
#         self.strde, self.padding = stride, padding
#
#         self.params = {'w': None, 'b': None}
#
#     def forward(self, inputs):
#         pass
#
#     def backward(self):
#         pass
#
#     def initialize(self):
#         self.params['w'] = w_init((self.h_f, self.w_f, self.n_f))
#         self.params['b'] = b_init((self.n_f, 1))
#
#     @property
#     def out_dim(self):
#         return (self.h_out, self.w_out, self.n_f)
#
#
# class Flatten(Layer):
#
#     def __init__(self, in_dim):
#         super().__init__('Flatten')
#         self._out_dim = np.prod(in_dim)
#
#     def forward(self, inputs):
#         self.input_shape = inputs.shape
#         out = inputs.ravel().reshape(self.input_shape[0], -1)
#         return out
#
#     def backward(self, grad):
#         grad = grad.reshape(self.input_shape)
#         return grad
#
#     @property
#     def out_dim(self):
#         return self._out_dim
#

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

    def initialize(self):
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

    def forward(self, inputs):
        if self.is_training:
            self._mask = np.random.binomial(1, self._keep_prob, size=inputs.shape) / self._keep_prob
            return inputs * self._mask
        else:
            return inputs

    def backward(self, grad):
        assert self.is_training is True
        return grad * self._mask

    def initialize(self):
        return
