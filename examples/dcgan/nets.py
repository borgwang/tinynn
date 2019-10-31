from tinynn.core.initializer import Normal
from tinynn.core.layer import Conv2D
from tinynn.core.layer import ConvTranspose2D
from tinynn.core.layer import Dense
from tinynn.core.layer import Flatten
from tinynn.core.layer import LeakyReLU
from tinynn.core.layer import MaxPool2D
from tinynn.core.layer import Reshape
from tinynn.core.layer import Sigmoid
from tinynn.core.net import Net


def G_mlp():
    w_init = Normal(0.0, 0.02)
    return Net([
        Dense(100, w_init=w_init),
        LeakyReLU(),
        Dense(300, w_init=w_init),
        LeakyReLU(),
        Dense(784, w_init=w_init),
        Sigmoid()])


def D_mlp():
    w_init = Normal(0.0, 0.02)
    return Net([
        Dense(300, w_init=w_init),
        LeakyReLU(),
        Dense(100, w_init=w_init),
        LeakyReLU(),
        Dense(1, w_init=w_init)])


def G_cnn():
    return Net([
        Dense(7 * 7 * 16),
        Reshape(7, 7, 16),
        ConvTranspose2D(kernel=[5, 5, 16, 6], stride=[2, 2], padding="SAME"),
        LeakyReLU(),
        ConvTranspose2D(kernel=[5, 5, 6, 1], stride=[2, 2], padding="SAME"),
        Sigmoid()])


def D_cnn():
    return Net([
        Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1], padding="SAME"),
        LeakyReLU(),
        MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        Conv2D(kernel=[5, 5, 6, 16], stride=[1, 1], padding="SAME"),
        LeakyReLU(),
        MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        Flatten(),
        Dense(120),
        LeakyReLU(),
        Dense(84),
        LeakyReLU(),
        Dense(1)])
