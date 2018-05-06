from typing import List

import numpy as np
from core.nn import NeuralNet
from core.tensor import Tensor


class Optimizer(object):

    def step(self, net: NeuralNet) -> None:
        # flatten all gradients
        grad = np.concatenate([np.ravel(grad) for param, grad in net.get_params_and_grads()])

        step = self._compute_step(grad)

        pointer = 0
        for param, grad in net.get_params_and_grads():
            block = np.prod(param.shape)
            param += step[pointer: pointer+block].reshape(param.shape)
            pointer += block

    def _compute_step(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr) -> None:
        self.lr = lr

    def _compute_step(self, grad: Tensor) -> Tensor:
        return - self.lr * grad


class Adam(Optimizer):

    def __init__(self,
                 lr,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8) -> None:
        self.lr = lr
        self._b1 = beta1
        self._b2 = beta2
        self._eps = epsilon

        self._t: int = 0

        self._m: Tensor = 0
        self._v: Tensor = 0

    def _compute_step(self, grad: Tensor) -> Tensor:
        self._t += 1

        lr_t = self.lr * np.sqrt(1 - np.power(self._b2, self._t)) / (1 - np.power(self._b1, self._t))

        self._m = self._b1 * self._m + (1 - self._b1) * grad
        self._v = self._b2 * self._v + (1 - self._b2) * np.square(grad)

        step = -lr_t * self._m / (np.sqrt(self._v) + self._eps)

        return step


class RMSProp(Optimizer):
    '''
    RMSProp maintain a moving (discouted) average of the square of gradients.
    Then divide gradients by the root of this average.

    mean_square = decay * mean_square{t-1} + (1-decay) * grad_t**2
    mom = momentum * mom{t-1} + lr * grad_t / sqrt(mean_square + epsilon)
    '''
    def __init__(self,
                 lr,
                 decay: float = 0.9,
                 momentum: float = 0.0,
                 epsilon: float = 1e-10) -> None:
        self.lr = lr
        self._decay = decay
        self._momentum = momentum
        self._eps = epsilon

        self._ms: Tensor = 0
        self._mom: Tensor = 0

    def _compute_step(self, grad: Tensor) -> Tensor:
        self._ms = self._decay * self._ms + (1 - self._decay) * np.square(grad)
        self._mom = self._momentum * self._mom + self.lr * grad / np.sqrt(self._ms + self._eps)

        step = -self._mom
        return step


class Momentum(Optimizer):
    # accumulation = momentum * accumulation + gradient
    # variable -= learning_rate * accumulation

    def __init__(self, lr, momentum: float = 0.9) -> None:
        self.lr = lr
        self._momentum = momentum

        self._acc: Tensor = 0

    def _compute_step(self, grad: Tensor) -> Tensor:
        self._acc = self._momentum * self._acc + grad
        step = - self.lr * self._acc
        return step
