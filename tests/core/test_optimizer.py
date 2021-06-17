import numpy as np
import pytest

from tinynn.core.optimizer import *
from tinynn.utils.seeder import random_seed
from tinynn.utils.structured_param import StructuredParam

lr = 0.1


@pytest.fixture(scope="function")
def fake_params():
    return StructuredParam([{"w": np.array([1., 2., 4.])}])


@pytest.fixture(scope="function")
def fake_grads():
    return StructuredParam([{"w": np.array([1., 2., 4.])}])


def test_sgd(fake_params, fake_grads):
    expect_params_values = fake_params.values - lr * fake_grads.values
    optimizer = SGD(lr)
    optimizer.step(fake_grads, fake_params)
    assert (fake_params.values == expect_params_values).all()


def test_momentum(fake_params, fake_grads):
    expect_params_values = fake_params.values - lr * fake_grads.values
    momentum = 0.9
    optimizer = Momentum(lr, momentum=momentum)
    optimizer.step(fake_grads, fake_params)
    assert (fake_params.values == expect_params_values).all()

    prev_acc = optimizer._acc
    grad_values = fake_grads.values
    optimizer.step(fake_grads, fake_params)
    expect_curr_acc = momentum * prev_acc + grad_values
    assert (optimizer._acc == expect_curr_acc).all()


def test_adam():
    pass


def test_adagrad():
    pass


def test_adadelta():
    pass


@pytest.fixture(scope="function")
def sgd_optimizer():
    return SGD(lr)


def test_scheduler_step_lr(sgd_optimizer):
    assert sgd_optimizer.lr == lr
    step_size = 10
    gamma = 0.9
    scheduler = StepLR(sgd_optimizer, step_size=step_size, gamma=gamma)
    for _ in range(step_size * 2):
        scheduler.step()
    assert sgd_optimizer.lr == lr * (gamma ** 2)


def test_scheduler_multi_step_lr(sgd_optimizer):
    assert sgd_optimizer.lr == lr
    milestones = [5]
    gamma = 0.9
    scheduler = MultiStepLR(sgd_optimizer, milestones=milestones, gamma=gamma)
    for _ in range(10):
        scheduler.step()
    assert sgd_optimizer.lr == lr * gamma


def test_scheduler_exponential_lr(sgd_optimizer):
    assert sgd_optimizer.lr == lr
    decay_steps = 1  # decay only in the first step
    decay_rate = 1. / np.e
    scheduler = ExponentialLR(sgd_optimizer, decay_steps=decay_steps, decay_rate=decay_rate)
    for _ in range(10):
        scheduler.step()
    assert sgd_optimizer.lr == lr * decay_rate ** (1. / 1.)


def test_scheduler_liner_lr(sgd_optimizer):
    assert sgd_optimizer.lr == lr
    decay_steps = 10
    final_lr = 0.0  # 0.1 -> 0.0 in 10 steps
    scheduler = LinearLR(sgd_optimizer, decay_steps=10)
    for _ in range(10):
        scheduler.step()
    assert np.abs(sgd_optimizer.lr - 0.) < 1e-5


def test_schedual_cyclical_lr(sgd_optimizer):
    assert sgd_optimizer.lr == lr
    min_lr, max_lr = 0.2, 0.3
    cyclical_steps = 20
    scheduler = CyclicalLR(sgd_optimizer, cyclical_steps, min_lr, max_lr)
    # lr step size
    lr_step_size = scheduler._abs_lr_delta
    assert lr_step_size == 2 * (max_lr - min_lr) / cyclical_steps

    # from init_lr to min_lr
    for _ in range(int((min_lr - lr) / lr_step_size)):
        scheduler.step()
    assert np.abs(sgd_optimizer.lr - min_lr) < 1e-5

    # reach max_lr after (cyclical_steps // 2) steps
    for _ in range(cyclical_steps // 2):
        scheduler.step()
    assert np.abs(sgd_optimizer.lr - max_lr) < 1e-5

    # reach min_lr after another (cyclical_steps // 2) steps
    for _ in range(cyclical_steps // 2):
        scheduler.step()
    assert np.abs(sgd_optimizer.lr - min_lr) < 1e-5
