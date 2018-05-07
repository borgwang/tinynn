from typing import Dict, Callable

import numpy as np

from core.tensor import Tensor
from core.initializer import Initializer, XavierNormalInit, ZerosInit


class Layer(object):

    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):

    def __init__(self,
                 num_in: int,
                 num_out: int,
                 w_init: Initializer = XavierNormalInit(),
                 b_init: Initializer = ZerosInit()) -> None:
        super().__init__()
        self.params['w'] = w_init((num_in, num_out))
        self.params['b'] = b_init((1, num_out))

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad: Tensor) -> Tensor:
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2


def log_softmax(x: Tensor) -> Tensor:
    exps = np.exp(x - np.max(x))
    return np.log(exps / np.sum(exps))


def log_softmax_prime(x: Tensor) -> Tensor:
    return log_softmax(x)


class Tanh(Activation):

    def __init__(self):
        super().__init__(tanh, tanh_prime)


class LogSoftmax(Activation):

    def __init__(self):
        super().__init__(log_softmax, log_softmax_prime)
