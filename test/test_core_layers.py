"""test unit for core/layers.py"""

import runtime_path  # isort:skip

import numpy as np

from core.layers import *


def test_dense():
    pass


def test_conv_2d():
    pass


def test_conv_transpose_2d():
    # test forward
    input_ = np.random.randn(256, 7, 7, 32)

    layer = ConvTranspose2D(
        kernel=[4, 4, 32, 64], stride=[1, 1], padding="SAME")
    output = layer.forward(input_)
    assert output.shape == (256, 7, 7, 64)

    layer = ConvTranspose2D(
        kernel=[4, 4, 32, 64], stride=[1, 1], padding="VALID")
    output = layer.forward(input_)
    assert output.shape == (256, 10, 10, 64)

    layer = ConvTranspose2D(
        kernel=[4, 4, 32, 64], stride=[3, 3], padding="SAME")
    output = layer.forward(input_)
    assert output.shape == (256, 21, 21, 64)

    layer = ConvTranspose2D(
        kernel=[4, 4, 32, 64], stride=[3, 3], padding="VALID")
    output = layer.forward(input_)
    assert output.shape == (256, 22, 22, 64)


def test_max_pool_2d():
    pass


def test_flatten():
    pass


def test_dropout():
    pass

