import numpy as np
import pytest
import tinynn as tn

tn.seeder.random_seed(31)
LR = 0.1


@pytest.fixture(name="params", scope="function")
def fixture_params():
    return tn.structured_param.StructuredParam([{"w": np.array([1., 2., 4.])}])


@pytest.fixture(name="grads", scope="function")
def fixture_grads():
    return tn.structured_param.StructuredParam([{"w": np.array([1., 2., 4.])}])


def test_step_fn(params, grads):
    origin_param_values, origin_grad_values = params.values, grads.values
    optim = tn.optimizer.SGD(LR, weight_decay=0.1)
    optim.step(grads, params)
    expect_grad_values = -LR * (origin_grad_values + 0.1 * origin_param_values)
    assert (grads.values == expect_grad_values).all()
    assert (params.values == origin_param_values + grads.values).all()


def test_sgd(grads):
    optim = tn.optimizer.SGD(LR)
    step = optim._compute_step(grads.values)
    assert (step == -LR * grads.values).all()


def test_momentum(grads):

    def _momentum_update(lr, mom, acc, grad):
        acc_t = mom * acc + grad
        step = -lr * acc_t
        return step, acc_t

    momentum = 0.9
    optim = tn.optimizer.Momentum(LR, momentum=momentum)
    for _ in range(3):
        step, acc_t = _momentum_update(LR, momentum, optim._acc, grads.values)
        actual_step = optim._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optim._acc.astype(float), acc_t.astype(float))


def test_adam(grads):

    def _adam_update(b1, b2, epsilon, m, v, lr, t, grad):
        t += 1
        lr_t = lr * np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        m_t = b1 * m + (1 - b1) * grad
        v_t = b2 * v + (1 - b2) * grad * grad
        step = - lr_t * m_t / (v_t ** 0.5 + epsilon)
        return step, m_t, v_t

    beta1, beta2, epsilon = 0.9, 0.99, 1e-8
    optim = tn.optimizer.Adam(LR, beta1=beta1, beta2=beta2, epsilon=epsilon)
    for _ in range(3):
        step, m_t, v_t = _adam_update(beta1, beta2, epsilon, optim._m, optim._v,
                                      LR, optim._t, grads.values)
        actual_step = optim._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optim._m.astype(float), m_t.astype(float))
        assert np.allclose(optim._v.astype(float), v_t.astype(float))


def test_radam(grads):

    def _radam_update(b1, b2, epsilon, m, v, lr, t, grad):
        t += 1
        rho = 2 / (1 - b2) - 1
        m_t = b1 * m + (1 - b1) * grad
        v_t = b2 * v + (1 - b2) * grad * grad
        m_t_ = m_t / (1 - b1 ** t)
        _rho_t = rho - 2 * b2 ** t / (1 - b2 ** t)
        if _rho_t > 4:
            v_t_ = v_t / (1 - b2 ** t)
            r_t = (((_rho_t - 4) * (_rho_t - 2) * rho) / \
                    ((rho - 4) * (rho - 2) * _rho_t)) ** 0.5
            step = -lr * m_t_ * r_t / (v_t_ ** 0.5 + epsilon)
        else:
            step = -lr * m_t_
        return step, m_t, v_t

    beta1, beta2, epsilon = 0.9, 0.99, 1e-8
    optim = tn.optimizer.RAdam(LR, beta1=beta1, beta2=beta2, epsilon=epsilon)
    for _ in range(3):
        step, m_t, v_t = _radam_update(beta1, beta2, epsilon, optim._m,
                                       optim._v, LR, optim._t, grads.values)
        actual_step = optim._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optim._m.astype(float), m_t.astype(float))
        assert np.allclose(optim._v.astype(float), v_t.astype(float))


def test_rmsprop(grads):

    def _rmsprop_update(lr, rho, momentum, epsilon, rms, mom, grad):
        rms_t = rho * rms + (1 - rho) * grad * grad
        mom_t = momentum * mom + lr * grad / ((rms_t + epsilon) ** 0.5)
        step = -mom_t
        return step, rms_t, mom_t

    decay, momentum, epsilon = 0.99, 0.9, 1e-8
    optim = tn.optimizer.RMSProp(LR, decay, momentum, epsilon)
    for _ in range(3):
        step, rms_t, mom_t = _rmsprop_update(
            LR, decay, momentum, epsilon, optim._rms, optim._mom, grads.values)
        actual_step = optim._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optim._rms.astype(float), rms_t.astype(float))
        assert np.allclose(optim._mom.astype(float), mom_t.astype(float))


def test_adagrad(grads):

    def _adagrad_update(lr, g, epsilon, grad):
        g_t = g + grad * grad
        step = -lr * grad / (g_t ** 0.5 + epsilon)
        return step, g_t

    epsilon = 1e-8
    optim = tn.optimizer.Adagrad(LR, epsilon=epsilon)
    for _ in range(3):
        step, g_t = _adagrad_update(LR, optim._g, epsilon, grads.values)
        actual_step = optim._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optim._g.astype(float), g_t.astype(float))


def test_adadelta(grads):

    def _adadelta_update(lr, rho, rms, delta, grad):
        rms_t = rho * rms + (1 - rho) * grad * grad
        d = (grad * (delta + epsilon) ** 0.5) / (rms_t + epsilon) ** 0.5
        delta_t = rho * delta + (1 - rho) * d * d
        step = -lr * d
        return step, rms_t, delta_t

    decay, epsilon = 0.9, 1e-8
    optim = tn.optimizer.Adadelta(LR, decay, epsilon)
    for _ in range(3):
        step, rms_t, delta_t = _adadelta_update(LR, decay, optim._rms,
                                                optim._delta, grads.values)
        actual_step = optim._compute_step(grads.values)
        assert np.allclose(step.astype(float), actual_step.astype(float))
        assert np.allclose(optim._rms.astype(float), rms_t.astype(float))
        assert np.allclose(optim._delta.astype(float), delta_t.astype(float))


@pytest.fixture(name="sgd", scope="function")
def fixture_sgd():
    return tn.optimizer.SGD(LR)


def test_scheduler_step_lr(sgd):
    step_size = 10
    gamma = 0.9
    scheduler = tn.optimizer.StepLR(sgd, step_size=step_size, gamma=gamma)
    for _ in range(step_size * 2):
        scheduler.step()
    assert sgd.lr == LR * (gamma ** 2)


def test_scheduler_multi_step_lr(sgd):
    milestones = [5]
    gamma = 0.9
    scheduler = tn.optimizer.MultiStepLR(
        sgd, milestones=milestones, gamma=gamma)
    for _ in range(10):
        scheduler.step()
    assert sgd.lr == LR * gamma


def test_scheduler_exponential_lr(sgd):
    decay_steps = 1  # decay only in the first step
    decay_rate = 1. / np.e
    scheduler = tn.optimizer.ExponentialLR(
        sgd, decay_steps=decay_steps, decay_rate=decay_rate)
    for _ in range(10):
        scheduler.step()
    assert sgd.lr == LR * decay_rate ** (1.0 / 1.0)


def test_scheduler_liner_lr(sgd):
    decay_steps = 10
    final_lr = 0.0  # 0.1 -> 0.0 in 10 steps
    scheduler = tn.optimizer.LinearLR(sgd, decay_steps=decay_steps)
    for _ in range(10):
        scheduler.step()
    assert np.abs(sgd.lr - final_lr) < 1e-5


def test_schedual_cyclical_lr(sgd):
    min_lr, max_lr = 0.2, 0.3
    cyclical_steps = 20
    scheduler = tn.optimizer.CyclicalLR(sgd, cyclical_steps, min_lr, max_lr)
    # lr step size
    lr_step_size = scheduler._abs_lr_delta
    assert lr_step_size == 2 * (max_lr - min_lr) / cyclical_steps

    # from init_lr to min_lr
    for _ in range(int((min_lr - LR) / lr_step_size)):
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
