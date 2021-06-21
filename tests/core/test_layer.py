import numpy as np
import pytest
import tinynn as tn

tn.seeder.random_seed(31)


@pytest.mark.parametrize("activation_layer, expect_range",
                         [(tn.layer.Sigmoid(), (0, 1)),
                          (tn.layer.Tanh(), (-1, 1)),
                          (tn.layer.ReLU(), (0, np.inf)),
                          (tn.layer.LeakyReLU(), (-np.inf, np.inf)),
                          (tn.layer.Softplus(), (0, np.inf))])
def test_activation(activation_layer, expect_range):
    """Test expected output range of activation layers"""
    input_ = np.random.normal(size=(100, 5))
    net = tn.net.Net([tn.layer.Dense(1), activation_layer])
    output = net.forward(input_)
    lower_bound, upper_bound = expect_range
    assert np.all((output >= lower_bound) & (output <= upper_bound))


def test_conv_transpose_2d():
    batch_size = 1
    input_ = np.random.randn(batch_size, 7, 7, 1)

    # test forward and backward correctness
    layer = tn.layer.ConvTranspose2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="VALID")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 22, 22, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape

    layer = tn.layer.ConvTranspose2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="SAME")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 21, 21, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape


def test_conv_2d():
    batch_size = 1
    input_ = np.random.randn(batch_size, 16, 16, 1)

    # test forward and backward correctness
    layer = tn.layer.Conv2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="VALID")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 5, 5, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape

    layer = tn.layer.Conv2D(
        kernel=[4, 4, 1, 2], stride=[3, 3], padding="SAME")
    output = layer.forward(input_)
    assert output.shape == (batch_size, 6, 6, 2)
    input_grads = layer.backward(output)
    assert input_grads.shape == input_.shape


def test_max_pool_2d():
    batch_size = 1
    channel = 2
    input_ = np.random.randn(batch_size, 4, 4, channel)

    layer = tn.layer.MaxPool2D(pool_size=[2, 2], stride=[2, 2])
    output = layer.forward(input_)
    assert output.shape == (batch_size, 2, 2, channel)

    layer = tn.layer.MaxPool2D(pool_size=[4, 4], stride=[2, 2])
    output = layer.forward(input_)
    answer = np.max(np.reshape(input_, (batch_size, -1, 2)), axis=1)
    assert (output.ravel() == answer.ravel()).all()


def test_reshape():
    batch_size = 1
    input_ = np.random.randn(batch_size, 2, 3, 4, 5)
    target_shape = (5, 4, 3, 2)
    layer = tn.layer.Reshape(*target_shape)
    output = layer.forward(input_)
    assert output.shape[1:] == target_shape


def test_rnn():
    batch_size = 1
    n_steps, input_dim = 10, 20
    input_ = np.random.randn(batch_size, n_steps, input_dim)
    layer = tn.layer.RNN(num_hidden=10, activation=tn.layer.Tanh())
    forward_out = layer.forward(input_)
    assert forward_out.shape == (batch_size, input_dim)

    fake_grads = np.random.randn(batch_size, input_dim)
    backward_out = layer.backward(fake_grads)
    # should has the same shape as input_
    assert backward_out.shape == (batch_size, n_steps, input_dim)


def test_batch_normalization():
    input_ = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                       [5.0, 4.0, 3.0, 2.0, 1.0]])
    mom, epsilon = 0.9, 1e-5
    layer = tn.layer.BatchNormalization(momentum=mom, epsilon=epsilon)
    for i in range(3):
        layer.forward(input_)
        mean = input_.mean(0, keepdims=True)
        var = input_.var(0, keepdims=True)
        if i == 0:
            r_mean = mean
            r_var = var
        else:
            r_mean = mom * r_mean + (1 - mom) * mean
            r_var = mom * r_var + (1 - mom) * var
        assert np.allclose(layer.ctx["X_norm"],
                           (input_ - mean) / (var + epsilon) ** 0.5)

    layer.is_training = False
    layer.forward(input_)
    assert np.allclose(layer.ctx["X_norm"],
                       (input_ - r_mean) / (r_var + epsilon) ** 0.5)


def test_dropout():
    batch_size, input_dim = 100, 1000
    input_ = np.ones((batch_size, input_dim))
    keep_prob = 0.5
    layer = tn.layer.Dropout(keep_prob=keep_prob)
    forward_out = layer.forward(input_)
    assert forward_out.shape == input_.shape
    keep_rate = 1. - (forward_out == 0.).sum() / (batch_size * input_dim)
    # varify keep_prob
    assert np.abs(keep_rate - keep_prob) < 1e-1
    # constent expectations
    assert np.abs(forward_out.mean() - input_.mean()) < 1e-1

    backward_out = layer.backward(input_)
    assert (backward_out == forward_out).all()

    layer.is_training = False
    forward_out = layer.forward(input_)
    assert (forward_out == input_).all()


def test_im2col():
    batch_size = 10
    input_ = np.random.randn(batch_size, 3, 3, 1)
    k_h, k_w, s_h, s_w = 2, 2, 1, 1
    output = tn.layer.im2col(input_, k_h, k_w, s_h, s_w)
    assert output.shape == (10 * 4, 4)
