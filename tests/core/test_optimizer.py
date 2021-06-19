import numpy as np
import pytest

from tinynn.core.optimizer import *
from tinynn.utils.seeder import random_seed
from tinynn.utils.structured_param import StructuredParam

random_seed(0)
lr = 0.1


@pytest.fixture(scope="function")
def params():
    return StructuredParam([{"w": np.array([1., 2., 4.])}])


@pytest.fixture(scope="function")
def grads():
    return StructuredParam([{"w": np.array([1., 2., 4.])}])


def test_step_fn(params, grads):
    origin_params_values, origin_grads_values = params.values, grads.values
    optimizer = SGD(lr, weight_decay=0.1)
    optimizer.step(grads, params)
    assert (grads.values == -lr * (origin_grads_values + 0.1 * origin_params_values)).all()
    assert (params.values == origin_params_values + grads.values).all()


def test_sgd(grads):
    optimizer = SGD(lr)
    step = optimizer._compute_step(grads.values)
    assert (step == -lr * grads.values).all()


def test_momentum(grads):

    def _momentum_update(lr, mom, acc, grad):
        acc_t = mom * acc + grad
        step = -lr * acc_t
        return step, acc_t

    momentum = 0.9
    optimizer = Momentum(lr, momentum=momentum)
    for _ in range(3):
        step, acc_t = _momentum_update(lr, momentum, optimizer._acc, grads.values)
        actual_step = optimizer._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optimizer._acc.astype(float), acc_t.astype(float))


def test_adam(grads):

    def _adam_update(beta1, beta2, epsilon, m, v, lr, t, grad):
        lr_t = lr * np.sqrt(1 - beta2**(t + 1)) / (1 - beta1**(t + 1))
        m_t = beta1 * m + (1 - beta1) * grad
        v_t = beta2 * v + (1 - beta2) * grad * grad
        return - lr_t * m_t / (v_t ** 0.5 + epsilon), m_t, v_t

    beta1, beta2, epsilon = 0.9, 0.99, 1e-8
    optimizer = Adam(lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
    for _ in range(3):
        step, m_t, v_t = _adam_update(beta1, beta2, epsilon, optimizer._m, optimizer._v,
                                      lr, optimizer._t, grads.values)
        actual_step = optimizer._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optimizer._m.astype(float), m_t.astype(float))
        assert np.allclose(optimizer._v.astype(float), v_t.astype(float))


def test_rmsprop(grads):

    def _rmsprop_update(lr, rho, momentum, epsilon, rms, mom, grad):
        rms_t = rho * rms + (1 - rho) * grad * grad
        mom_t = momentum * mom + lr * grad / ((rms_t + epsilon) ** 0.5)
        step = -mom_t
        return step, rms_t, mom_t

    decay, momentum, epsilon = 0.99, 0.9, 1e-8
    optimizer = RMSProp(lr, decay, momentum, epsilon)
    for _ in range(3):
        step, rms_t, mom_t = _rmsprop_update(lr, decay, momentum, epsilon, optimizer._rms,
                                             optimizer._mom, grads.values)
        actual_step = optimizer._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optimizer._rms.astype(float), rms_t.astype(float))
        assert np.allclose(optimizer._mom.astype(float), mom_t.astype(float))


def test_adagrad(grads):

    def _adagrad_update(lr, g, epsilon, grad):
        g_t = g + grad * grad
        step = -lr * grad / (g_t ** 0.5 + epsilon)
        return step, g_t

    epsilon = 1e-8
    optimizer = Adagrad(lr, epsilon=epsilon)
    for _ in range(3):
        step, g_t = _adagrad_update(lr, optimizer._g, epsilon, grads.values)
        actual_step = optimizer._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optimizer._g.astype(float), g_t.astype(float))


def test_adadelta(grads):

    def _adadelta_update(lr, rho, rms, delta, grad):
        rms_t = rho * rms + (1 - rho) * grad * grad
        d = (grad * (delta + epsilon) ** 0.5) / (rms_t + epsilon) ** 0.5
        delta_t = rho * delta + (1 - rho) * d * d
        step = -lr * d
        return step, rms_t, delta_t

    decay, epsilon = 0.9, 1e-8
    optimizer = Adadelta(lr, decay, epsilon)
    for _ in range(3):
        step, rms_t, delta_t = _adadelta_update(lr, decay, optimizer._rms, optimizer._delta, 
                                                grads.values)
        actual_step = optimizer._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optimizer._rms.astype(float), rms_t.astype(float))
        assert np.allclose(optimizer._delta.astype(float), delta_t.astype(float))


@pytest.fixture(scope="function")
def sgd():
    return SGD(lr)


def test_scheduler_step_lr(sgd):
    assert sgd.lr == lr
    step_size = 10
    gamma = 0.9
    scheduler = StepLR(sgd, step_size=step_size, gamma=gamma)
    for _ in range(step_size * 2):
        scheduler.step()
    assert sgd.lr == lr * (gamma ** 2)


def test_scheduler_multi_step_lr(sgd):
    assert sgd.lr == lr
    milestones = [5]
    gamma = 0.9
    scheduler = MultiStepLR(sgd, milestones=milestones, gamma=gamma)
    for _ in range(10):
        scheduler.step()
    assert sgd.lr == lr * gamma


def test_scheduler_exponential_lr(sgd):
    assert sgd.lr == lr
    decay_steps = 1  # decay only in the first step
    decay_rate = 1. / np.e
    scheduler = ExponentialLR(sgd, decay_steps=decay_steps, decay_rate=decay_rate)
    for _ in range(10):
        scheduler.step()
    assert sgd.lr == lr * decay_rate ** (1. / 1.)


def test_scheduler_liner_lr(sgd):
    assert sgd.lr == lr
    decay_steps = 10
    final_lr = 0.0  # 0.1 -> 0.0 in 10 steps
    scheduler = LinearLR(sgd, decay_steps=10)
    for _ in range(10):
        scheduler.step()
    assert np.abs(sgd.lr - 0.) < 1e-5


def test_schedual_cyclical_lr(sgd):
    assert sgd.lr == lr
    min_lr, max_lr = 0.2, 0.3
    cyclical_steps = 20
    scheduler = CyclicalLR(sgd, cyclical_steps, min_lr, max_lr)
    # lr step size
    lr_step_size = scheduler._abs_lr_delta
    assert lr_step_size == 2 * (max_lr - min_lr) / cyclical_steps

    # from init_lr to min_lr
    for _ in range(int((min_lr - lr) / lr_step_size)):
        scheduler.step()
    assert np.abs(sgd.lr - min_lr) < 1e-5

    # reach max_lr after (cyclical_steps // 2) steps
    for _ in range(cyclical_steps // 2):
        scheduler.step()
    assert np.abs(sgd.lr - max_lr) < 1e-5

    # reach min_lr after another (cyclical_steps // 2) steps
    for _ in range(cyclical_steps // 2):
        scheduler.step()
    assert np.abs(sgd.lr - min_lr) < 1e-5
