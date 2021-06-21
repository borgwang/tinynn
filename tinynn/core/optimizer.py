"""Various optimization algorithms and learning rate schedulers."""

import numpy as np


class Optimizer:

    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, grads, params):
        # compute the gradient step
        grads = self.compute_step(grads)
        # apply weight_decay if specified
        if self.weight_decay:
            grads -= self.lr * self.weight_decay * params
        # take a step
        params += grads

    def compute_step(self, grads):
        grads.values = self._compute_step(grads.values)
        return grads

    def _compute_step(self, grads):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr, weight_decay)

    def _compute_step(self, grads):
        return -self.lr * grads


class Adam(Optimizer):

    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._b1 = beta1
        self._b2 = beta2
        self._epsilon = epsilon

        self._t = 0
        self._m = 0
        self._v = 0

    def _compute_step(self, grads):
        self._t += 1

        self._m += (1.0 - self._b1) * (grads - self._m)
        self._v += (1.0 - self._b2) * (grads ** 2 - self._v)

        # bias correction
        _m = self._m / (1 - self._b1 ** self._t)
        _v = self._v / (1 - self._b2 ** self._t)

        step = -self.lr * _m / (_v ** 0.5 + self._epsilon)
        return step


class RAdam(Optimizer):
    """Rectified Adam. Ref: https://arxiv.org/pdf/1908.03265v1.pdf """
    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._b1 = beta1
        self._b2 = beta2
        self._epsilon = epsilon

        self._t = 0
        self._m = 0
        self._v = 0

        self.rho = 2.0 / (1 - self._b2) - 1.0

    def _compute_step(self, grads):
        self._t += 1

        self._m += (1.0 - self._b1) * (grads - self._m)
        self._v += (1.0 - self._b2) * (grads ** 2 - self._v)

        # bias correction
        _m = self._m / (1 - self._b1 ** self._t)

        _rho = self.rho - 2 * self._b2 ** self._t / (1 - self._b2 ** self._t)
        if _rho > 4.0:
            _v = self._v / (1 - self._b2 ** self._t)
            _r = (((_rho - 4) * (_rho - 2) * self.rho) / \
                    ((self.rho - 4) * (self.rho - 2) * _rho)) ** 0.5
            step = -self.lr * _m * _r / (_v ** 0.5 + self._epsilon)
        else:
            step = -self.lr * _m
        return step


class RMSProp(Optimizer):
    """Root Mean Square Prop optimizer
    mean_square = decay * mean_square{t-1} + (1-decay) * grad_t**2
    mom = momentum * mom{t-1} + lr * grad_t / sqrt(mean_square + epsilon)
    """
    def __init__(self,
                 lr=0.01,
                 decay=0.99,
                 momentum=0.0,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._rho = decay
        self._momentum = momentum
        self._epsilon = epsilon

        self._rms = 0
        self._mom = 0

    def _compute_step(self, grads):
        self._rms += (1 - self._rho) * (grads ** 2 - self._rms)
        self._mom = self._momentum * self._mom + self.lr * grads / \
                (self._rms + self._epsilon) ** 0.5
        step = -self._mom
        return step


class Momentum(Optimizer):
    """accumulation = momentum * accumulation + gradient
    variable -= learning_rate * accumulation
    """
    def __init__(self, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._momentum = momentum
        self._acc = 0

    def _compute_step(self, grads):
        self._acc = self._momentum * self._acc + grads
        step = -self.lr * self._acc
        return step


class Adagrad(Optimizer):
    """AdaGrad optimizer
    accumulation = - (learning_rate / sqrt(G + epsilon)) * gradient
    where G is the element-wise sum of square gradient
    ref: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    def __init__(self, lr, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._g = 0
        self._epsilon = epsilon

    def _compute_step(self, grads):
        self._g += grads ** 2
        adjust_lr = self.lr / (self._g + self._epsilon) ** 0.5
        step = -adjust_lr * grads
        return step


class Adadelta(Optimizer):
    """Adadelta algorithm (https://arxiv.org/abs/1212.5701)"""
    def __init__(self, lr=1.0, decay=0.9, epsilon=1e-8, weight_decay=0.0,):
        super().__init__(lr, weight_decay)
        self._epsilon = epsilon
        self._rho = decay
        self._rms = 0  # running average of square gradient
        self._delta = 0  # running average of delta

    def _compute_step(self, grads):
        self._rms += (1 - self._rho) * (grads ** 2 - self._rms)
        std = (self._delta + self._epsilon) ** 0.5
        delta = grads * (std / (self._rms + self._epsilon) ** 0.5)
        step = - self.lr * delta
        self._delta += (1 - self._rho) * (delta ** 2 - self._delta)
        return step


class BaseScheduler:
    """BaseScheduler model receive a optimizer and Adjust the lr
    by calling step() method during training.
    """
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._init_lr = self.curr_lr

        self._t = 0

    def step(self):
        self._t += 1
        self._optimizer.lr = self._compute_lr()
        return self.curr_lr

    def _compute_lr(self):
        raise NotImplementedError

    @property
    def curr_lr(self):
        return self._optimizer.lr


class StepLR(BaseScheduler):
    """LR decayed by gamma every "step_size" epochs."""
    def __init__(self,
                 optimizer,
                 step_size,
                 gamma=0.1):
        super().__init__(optimizer)
        assert step_size >= 1
        self._step_size = step_size
        self._gamma = gamma

    def _compute_lr(self):
        decay = self._gamma if self._t % self._step_size == 0 else 1.0
        return decay * self.curr_lr


class MultiStepLR(BaseScheduler):
    """LR decayed by gamma when #steps reaches one of the milestones.
    Milestones must be monotonically increasing.
    """
    def __init__(self, optimizer, milestones, gamma=0.1):
        super().__init__(optimizer)
        milestones = [int(m) for m in milestones]
        assert len(milestones) > 0, "milestones requires at-least one element!"
        assert all(x < y for x, y in zip(milestones[:-1], milestones[1:])), \
               "milestones must be a list of int and be increasing!"

        self._milestones = milestones
        self._gamma = gamma

    def _compute_lr(self):
        decay = self._gamma if self._t in self._milestones else 1.0
        return decay * self.curr_lr


class ExponentialLR(BaseScheduler):
    """ExponentialLR is computed by:
    lr_decayed = lr * decay_rate ^ (current_steps / decay_steps)
    """
    def __init__(self,
                 optimizer,
                 decay_steps,
                 decay_rate=(1. / np.e)):
        super().__init__(optimizer)
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate

    def _compute_lr(self):
        if self._t <= self._decay_steps:
            decay = self._decay_rate ** (self._t / self._decay_steps)
            return self._init_lr * decay
        return self.curr_lr


class LinearLR(BaseScheduler):
    """Linear decay learning rate when the number of the epoch is in
    [start_step, start_step + decay_steps]
    """
    def __init__(self,
                 optimizer,
                 decay_steps,
                 final_lr=1e-6,
                 start_step=0):
        super().__init__(optimizer)
        assert decay_steps > 0

        self._lr_delta = (final_lr - self._init_lr) / decay_steps

        self._final_lr = final_lr
        self._decay_steps = decay_steps
        self._start_step = start_step

    def _compute_lr(self):
        if self._t > self._start_step:
            if self._t <= self._start_step + self._decay_steps:
                return self.curr_lr + self._lr_delta
        return self.curr_lr


class CyclicalLR(BaseScheduler):
    """Cyclical increase and decrease learning rate within a reasonable range.
    Ref: https://arxiv.org/pdf/1506.01186.pdf
    """
    def __init__(self,
                 optimizer,
                 cyclical_steps,
                 min_lr=1e-3,
                 max_lr=1e-2):
        super().__init__(optimizer)
        assert cyclical_steps > 2
        assert max_lr >= min_lr
        self._cyclical_steps = cyclical_steps
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._abs_lr_delta = 2 * (max_lr - min_lr) / cyclical_steps

        self._is_cycling = False
        self._cycling_start_t = None

    def _compute_lr(self):
        if self.curr_lr > self._max_lr:
            return self.curr_lr - self._abs_lr_delta
        if self.curr_lr < self._min_lr:
            return self.curr_lr + self._abs_lr_delta

        if not self._is_cycling:
            self._is_cycling = True
            self._cycling_start_t = self._t

        if ((self._t - self._cycling_start_t) % self._cyclical_steps <
                self._cyclical_steps // 2):
            return self.curr_lr + self._abs_lr_delta
        return self.curr_lr - self._abs_lr_delta
