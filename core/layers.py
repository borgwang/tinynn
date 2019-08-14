# Author: borgwang <borgwang@126.com>
# Date: 2018-05-05
#
# Filename: layers.py
# Description: Network layers and Activation layers...

import numpy as np

from core.initializer import XavierNormalInit
from core.initializer import ZerosInit


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
        self.is_training = True if phase == "TRAIN" else False

    @property
    def out_dim(self):
        pass


# ----------
# Network Layers
# ----------

class Linear(Layer):

    def __init__(self,
                 num_in,
                 num_out,
                 w_init=XavierNormalInit(),
                 b_init=ZerosInit()):
        super().__init__("Linear")
        # self.params["w"] = w_init((num_in, num_out))
        # self.params["b"] = b_init((1, num_out))
        self.w_shape = (num_in, num_out)
        self.b_shape = (1, num_out)
        self.w_init = w_init
        self.b_init = b_init
        self.shape = [self.w_shape, self.b_shape]

        self.params = {"w": None, "b": None}
        self.is_init = False

    def forward(self, inputs):
        if not self.is_init:
            raise ValueError("Parameters uninitialized error!")
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T

    def initialize(self):
        self.params["w"] = self.w_init(self.w_shape)
        self.params["b"] = self.b_init(self.b_shape)
        self.is_init = True


class Conv2D(Layer):

    def __init__(self,
                 kernel,
                 stride,
                 padding="SAME",
                 w_init=XavierNormalInit(),
                 b_init=ZerosInit()):
        """
        :param kernel: A list of int that has length 4 (height, width, in_channels, out_channels)
        :param stride: A list of int that has length 2 (height, width)
        :param padding: String ["SAME", "VALID", "FULL"]
        :param w_init: weight initializer
        :param b_init: bias initializer
        """
        super().__init__("Linear")

        # verify arguments
        assert len(kernel) == 4
        assert len(stride) == 2
        assert padding in ["FULL", "SAME", "VALID"]

        self.padding_mode = padding
        self.kernel = kernel
        self.stride = stride
        self.w_init = w_init
        self.b_init = b_init

        self.params = {"w": None, "b": None}
        self.is_init = False

    def forward(self, inputs):
        ks = self.kernel[:2]  # kernel size
        pad = self._get_padding(ks, self.padding_mode)
        pad_width = ((0, 0), (pad[0], pad[1]), (pad[2], pad[3]), (0, 0))
        padded = np.pad(inputs, pad_width=pad_width, mode="constant")

        in_n, in_h, in_w, in_c = inputs.shape
        out_h = int((in_h + pad[0] + pad[1] - ks[0]) / self.stride[0] + 1)
        out_w = int((in_w + pad[2] + pad[3] - ks[1]) / self.stride[1] + 1)
        out_c = self.kernel[-1]

        kernel = np.repeat(self.params["w"][np.newaxis, :, :, :, :], in_n, axis=0)
        outputs = np.empty(shape=(in_n, out_h, out_w, out_c))
        for i, col in enumerate(range(0, padded.shape[1] - ks[0] + 1, self.stride[0])):
            for j, row in enumerate(range(0, padded.shape[2] - ks[1] + 1, self.stride[1])):
                patch = padded[:, col:col+ks[0], row:row+ks[1], :]
                patch = np.repeat(patch[:, :, :, :, np.newaxis], out_c, axis=-1)
                outputs[:, i, j] = np.sum(patch * kernel, axis=(1, 2, 3))
        outputs += self.params["b"]
        return outputs

    def backward(self, grad):
        pass

    def initialize(self):
        self.params["w"] = self.w_init(shape=self.kernel)
        self.params["b"] = self.b_init(shape=self.kernel[-1])
        self.is_init = True

    @staticmethod
    def _get_padding(ks, mode):
        """
        params: ks (kernel size) [p, q]
        return: list of padding (top, bottom, left, right) in different modes
        """
        pad = None
        if mode == "FULL":
            pad = [ks[0] - 1, ks[1] - 1, ks[0] - 1, ks[1] - 1]
        elif mode == "VALID":
            pad = [0, 0, 0, 0]
        elif mode == "SAME":
            pad = [(ks[0] - 1) // 2, (ks[0] - 1) // 2,
                   (ks[1] - 1) // 2, (ks[1] - 1) // 2]
            if ks[0] % 2 == 0:
                pad[1] += 1
            if ks[1] % 2 == 0:
                pad[3] += 1
        else:
            print("Invalid mode")
        return pad

# class Flatten(Layer):
#
#     def __init__(self, in_dim):
#         super().__init__("Flatten")
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

    def __init__(self, name):
        super().__init__(name)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative_func(self.inputs) * grad

    def initialize(self):
        return

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, x):
        raise NotImplementedError


class Sigmoid(Activation):

    def __init__(self):
        super().__init__("Sigmoid")

    def func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_func(self, x):
        return self.func(x) * (1.0 - self.func(x))


class Tanh(Activation):

    def __init__(self):
        super().__init__("Tanh")

    def func(self, x):
        return np.tanh(x)

    def derivative_func(self, x):
        return 1.0 - self.func(x) ** 2


class ReLU(Activation):

    def __init__(self):
        super().__init__("ReLU")

    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative_func(self, x):
        return x > 0.0


class LeakyReLU(Activation):

    def __init__(self, slope=0.01):
        super().__init__("LeakyReLU")
        self._slope = slope

    def func(self, x):
        # TODO: maybe a litter bit slow due to the copy
        x = x.copy()
        x[x < 0.0] *= self._slope
        return x

    def derivative_func(self, x):
        x[x < 0.0] = self._slope
        return x


class Dropout(Layer):

    def __init__(self, keep_prob=0.5):
        super().__init__("Dropout")
        self._keep_prob = keep_prob
        self._multiplier = None

    def forward(self, inputs):
        if self.is_training:
            multiplier = np.random.binomial(
                1, self._keep_prob, size=inputs.shape)
            self._multiplier = multiplier / self._keep_prob
            return inputs * self._multiplier
        else:
            return inputs

    def backward(self, grad):
        assert self.is_training is True
        return grad * self._multiplier

    def initialize(self):
        return
