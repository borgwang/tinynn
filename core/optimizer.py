from typing import List

import numpy as np
from core.nn import NeuralNet
from core.tensor import Tensor


class Optimizer(object):

    def __init__(self, lr):
        self.lr = lr

    def step(self, net):
        # flatten all gradients
        grad = np.concatenate(
            [np.ravel(grad) for param, grad in net.get_params_and_grads()])

        step = self._compute_step(grad)

        pointer = 0
        for param, grad in net.get_params_and_grads():
            block = np.prod(param.shape)
            param += step[pointer: pointer+block].reshape(param.shape)
            pointer += block

    def _compute_step(self, grad):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr):
        super().__init__(lr)

    def _compute_step(self, grad):
        return - self.lr * grad


class Adam(Optimizer):

    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8):
        super().__init__(lr)
        self._b1 = beta1
        self._b2 = beta2
        self._eps = epsilon

        self._t= 0

        self._m= 0
        self._v= 0

    def _compute_step(self, grad):
        self._t += 1

        lr_t = self.lr * np.sqrt(1 - np.power(self._b2, self._t)) / \
            (1 - np.power(self._b1, self._t))

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
                 lr=0.01,
                 decay=0.99,
                 momentum=0.0,
                 epsilon=1e-8):
        super().__init__(lr)
        self._decay = decay
        self._momentum = momentum
        self._eps = epsilon

        self._ms: Tensor = 0
        self._mom: Tensor = 0

    def _compute_step(self, grad):
        self._ms = self._decay * self._ms + (1 - self._decay) * np.square(grad)
        self._mom = self._momentum * self._mom + \
            self.lr * grad / np.sqrt(self._ms + self._eps)

        step = -self._mom
        return step


class Momentum(Optimizer):
    '''
     accumulation = momentum * accumulation + gradient
     variable -= learning_rate * accumulation
    '''
    def __init__(self, lr, momentum=0.9):
        super().__init__(lr)
        self._momentum = momentum
        self._acc: Tensor = 0

    def _compute_step(self, grad):
        self._acc = self._momentum * self._acc + grad
        step: Tensor = -self.lr * self._acc
        return step


class LRScheduler(object):
    '''
    LRScheduler model receive a optimizer and Adjust the lr by calling
    step() method during training.
    '''
    def __init__(self, optimizer):
        self._optim = optimizer
        self._initial_lr = self.get_current_lr()

        self._t: int = 0

    def step(self):
        self._t += 1
        self._optim.lr = self._compute_lr()
        return self.get_current_lr()

    def _compute_lr(self):
        raise NotImplementedError

    def get_current_lr(self):
        return self._optim.lr


class StepLR(LRScheduler):
    '''
    LR decayed by gamma every 'step_size' epoches.
    '''
    def __init__(self,
                 optimizer,
                 step_size,
                 gamma=0.1):
        super().__init__(optimizer)
        assert step_size >= 1, 'step_size must greater than 0 (%d was set)' % step_size
        self._step_size = step_size
        self._gamma = gamma

    def _compute_lr(self):
        decay = self._gamma if self._t % self._step_size == 0 else 1.0
        return decay * self.get_current_lr()


class MultiStepLR(LRScheduler):
    '''
    LR decayed by gamma when the number of epoch reaches one of the milestones.
    Argument 'milestones' must be a int list and be increasing.
    '''
    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1):
        super().__init__(optimizer)
        milestones = [int(m) for m in milestones]
        assert all(x < y for x, y in zip(milestones[:-1], milestones[1:])) and \
            all(isinstance(x, int) for x in milestones), \
            'milestones must be a list of int and be increasing!'

        self._milestones = milestones
        self._gamma = gamma

    def _compute_lr(self):
        decay = self._gamma if self._t in self._milestones else 1.0
        return decay * self.get_current_lr()


class ExponentialLR(LRScheduler):
    '''
    ExponentialLR is computed by:

    lr_decayed = lr * decay_rate ^ (current_steps / decay_steps)
    '''
    def __init__(self,
                 optimizer,
                 decay_steps,
                 decay_rate=(1 / np.e)):
        super().__init__(optimizer)
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate

    def _compute_lr(self):
        if self._t <= self._decay_steps:
            return self._initial_lr * \
                self._decay_rate ** (self._t  / self._decay_steps)
        else:
            return self.get_current_lr()


class LinearLR(LRScheduler):
    '''
    Linear decay learning rate when the number of the epoche is in
    [start_step, start_step + decay_steps]
    '''
    def __init__(self,
                 optimizer,
                 decay_steps,
                 final_lr=1e-6,
                 start_step=0):
        super().__init__(optimizer)
        assert final_lr < self._initial_lr, \
            'The final lr should be no greater than the initial lr.'
        assert decay_steps > 0

        self._lr_delta = (final_lr - self._initial_lr) / decay_steps

        self._final_lr = final_lr
        self._decay_steps = decay_steps
        self._start_step = start_step

    def _compute_lr(self):
        if self._t > self._start_step:
            if self._t <= self._start_step + self._decay_steps:
                return self.get_current_lr() + self._lr_delta
        return self.get_current_lr()
