from typing import Tuple

import numpy as np

from core.tensor import Tensor


class Initializer(object):
    def __call__(self, shape: Tuple) -> Tensor:
        raise NotImplementedError


class NormalInit(Initializer):

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self._mean = mean
        self._std = std

    def __call__(self, shape: Tuple) -> Tensor:
        return np.random.normal(loc=self._mean, scale=self._std, size=shape)


class UniformInit(Initializer):

    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        self._a = a
        self._b = b

    def __call__(self, shape: Tuple) -> Tensor:
        return np.random.uniform(low=self._a, high=self._b, size=shape)


class ZerosInit(Initializer):

    def __call__(self, shape: Tuple) -> Tensor:
        return np.zeros(shape=shape, dtype=float)


class ConstantInit(Initializer):

    def __init__(self, val : float) -> None:
        self._val = val

    def __call__(self, shape: Tuple) -> Tensor:
        return np.full(shape=shape, fill_value=self._val, dtype=float)


class XavierUniformInit(Initializer):
    '''
    Implement the Xavier method descripted in
    "Understanding the difficulty of training deep feedforward neural networks”
    Glorot, X. & Bengio, Y. (2010)

    Weights will have values sampled from uniform distribution U(-a, a) where
    a = gain * sqrt(6.0 / (num_in + num_out))

    '''
    def __init__(self, gain: float = 1.0) -> None:
        self._gain = gain

    def __call__(self, shape: Tuple) -> Tensor:
        assert len(shape) >= 2
        a = self._gain * np.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(low=-a, high=a, size=shape)


class XavierNormalInit(Initializer):
    '''
    Implement the Xavier method descripted in
    "Understanding the difficulty of training deep feedforward neural networks”
    Glorot, X. & Bengio, Y. (2010)

    Weights will have values sampled from uniform distribution N(0, std) where
    std = gain * sqrt(1.0 / (num_in + num_out))
    '''
    def __init__(self, gain: float = 1.0) -> None:
        self._gain = gain

    def __call__(self, shape: Tuple) -> Tensor:
        assert len(shape) >= 2
        std = self._gain * np.sqrt(2.0 / (shape[0] + shape[1]))
        return np.random.normal(loc=0.0, scale=std, size=shape)


class HeUniformInit(Initializer):
    '''
    Implement the He initialization method descripted in
    “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification”
    He, K. et al. (2015)

    Weights will have values sampled from uniform distribution U(-a, a) where
    a = sqrt(6.0 / num_in)
    '''
    def __init__(self, gain: float = 1.0) -> None:
        self._gain = gain

    def __call__(self, shape: Tuple) -> Tensor:
        a = self._gain * np.sqrt(6.0 / shape[0])
        return np.random.uniform(low=-a, high=a, size=shape)


class HeNormalInit(Initializer):
    '''
    Implement the He initialization method descripted in
    “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification”
    He, K. et al. (2015)

    Weights will have values sampled from normal distribution N(0, std) where
    std = sqrt(2.0 / num_in)
    '''
    def __init__(self, gain: float = 1.0) -> None:
        self._gain = gain

    def __call__(self, shape: Tuple) -> Tensor:
        std = self._gain * np.sqrt(2.0 / shape[0])
        return np.random.normal(loc=0.0, scale=std, size=shape)


class OrthogonalInit(Initializer):
    '''
    Implement the initialization method descripted in
    “Exact solutions to the nonlinear dynamics of learning in deep linear neural networks”
    Saxe, A. et al. (2013)

    The shape must be at least 2 dimensional.
    '''
    def __init__(self, gain: float = 1.0) -> None:
        self._gain = gain

    def __call__(self, shape: Tuple) -> Tensor:
        assert len(shape) == 2  # only support 2 dimension tensor for now
        a = np.random.normal(0.0, 1.0, shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else v
        q = q.reshape(shape)
        return (self._gain * q[:shape[0], :shape[1]]).astype(np.float32)


class SparseInit(Initializer):
    '''
    Implement the initialization method descripted in
    “Deep learning via Hessian-free optimization” - Martens, J. (2010).

    Weights will be initialized as a sparse tensor. Non-zero elements will
    have values sampled from normal distribution N(0, 0.01)
    '''
    def __init__(self, sparsity: float = 0.1, std: float = 0.01) -> None:
        self._sparsity = sparsity
        self._std = std

    def __call__(self, shape: Tuple) -> Tensor:
        assert len(shape) == 2
        n = np.prod(shape)
        num_zeros = int(np.ceil(n * self._sparsity))
        mask = np.full(n, True)
        mask[:num_zeros] = False
        np.random.shuffle(mask)
        mask = mask.reshape(shape)
        return np.random.normal(loc=0.0, scale=self._std, size=shape) * mask
