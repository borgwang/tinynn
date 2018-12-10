# Author: borgwang <borgwang@126.com>
# Date: 2018-05-05
#
# Filename: BaseOptimizer.py
# Description:
#   Implement multiple optimization algorithms and learning rate scheduler.


import numpy as np


# ----------
# Optimizer
# ----------

class BaseOptimizer(object):

    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def compute_step(self, grads, params):
        step = list()
        # flatten all gradients
        flatten_grads = np.concatenate(
            [np.ravel(v) for grad in grads for v in grad.values()])
        # compute step
        flatten_step = self._compute_step(flatten_grads)

        p = 0
        for param in params:
            layer = dict()
            for k, v in param.items():
                block = np.prod(v.shape)
                _step = flatten_step[p:p+block].reshape(v.shape)
                _step -= self.weight_decay * v
                layer[k] = _step
                p += block
            step.append(layer)
        return step

    def _compute_step(self, grad):
        raise NotImplementedError


class SGD(BaseOptimizer):

    def __init__(self, lr, weight_decay=0.0):
        super().__init__(lr, weight_decay)

    def _compute_step(self, grad):
        return - self.lr * grad


class Adam(BaseOptimizer):

    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._b1 = beta1
        self._b2 = beta2
        self._eps = eps

        self._t= 0
        self._m= 0
        self._v= 0

    def _compute_step(self, grad):
        self._t += 1

        lr_t = self.lr * (1 - self._b2 ** self._t) ** 0.5 / \
            (1 - self._b1 ** self._t)

        self._m = self._b1 * self._m + (1 - self._b1) * grad
        self._v = self._b2 * self._v + (1 - self._b2) * (grad ** 2)

        step = -lr_t * self._m / ((self._v ** 0.5) + self._eps)

        return step


class RMSProp(BaseOptimizer):
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
                 eps=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._decay = decay
        self._momentum = momentum
        self._eps = eps

        self._ms = 0
        self._mom = 0

    def _compute_step(self, grad):
        self._ms = self._decay * self._ms + (1 - self._decay) * np.square(grad)
        self._mom = self._momentum * self._mom + \
            self.lr * grad / np.sqrt(self._ms + self._eps)

        step = -self._mom
        return step


class Momentum(BaseOptimizer):
    '''
     accumulation = momentum * accumulation + gradient
     variable -= learning_rate * accumulation
    '''
    def __init__(self, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._momentum = momentum
        self._acc = 0

    def _compute_step(self, grad):
        self._acc = self._momentum * self._acc + grad
        step = -self.lr * self._acc
        return step


# ----------
# Learning Rate Scheduler
# ----------

class BaseScheduler(object):
    '''
    BaseScheduler model receive a optimizer and Adjust the lr by calling
    step() method during training.
    '''
    def __init__(self, optimizer):
        self._optim = optimizer
        self._initial_lr = self.curr_lr

        self._t = 0

    def step(self):
        self._t += 1
        self._optim.lr = self._compute_lr()
        return self.curr_lr

    def _compute_lr(self):
        raise NotImplementedError

    @property
    def curr_lr(self):
        return self._optim.lr


class StepLR(BaseScheduler):
    '''
    LR decayed by gamma every 'step_size' epochs.
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
        return decay * self.curr_lr


class MultiStepLR(BaseScheduler):
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
        return decay * self.curr_lr


class ExponentialLR(BaseScheduler):
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
            return self.curr_lr


class LinearLR(BaseScheduler):
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
        assert decay_steps > 0

        self._lr_delta = (final_lr - self._initial_lr) / decay_steps

        self._final_lr = final_lr
        self._decay_steps = decay_steps
        self._start_step = start_step

    def _compute_lr(self):
        if self._t > self._start_step:
            if self._t <= self._start_step + self._decay_steps:
                return self.curr_lr + self._lr_delta
        return self.curr_lr


class CyclicalLR(BaseScheduler):
    '''
    Cyclical increase and decrease learning rate within a reasonable range.
    See https://arxiv.org/pdf/1506.01186.pdf for details.
    '''
    def __init__(self,
                 optimizer,
                 cyclical_steps,
                 max_lr=1e-2,
                 min_lr=1e-3):
        super().__init__(optimizer)
        assert cyclical_steps > 2
        assert max_lr >= min_lr
        self._cyclical_steps = cyclical_steps
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._abs_lr_delta = 2 * (max_lr - min_lr) / cyclical_steps

    def _compute_lr(self):
        if self._t % self._cyclical_steps < self._cyclical_steps // 2:
            return self.curr_lr + self._abs_lr_delta
        else:
            return self.curr_lr - self._abs_lr_delta
