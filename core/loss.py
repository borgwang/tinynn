import numpy as np

from core.tensor import Tensor


class Loss(object):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSELoss(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)


class SparseCrossEntropyLoss(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        pass

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        pass


class CrossEntropyLoss(Loss):
    '''
    L = weight[class] * (-log(exp(x[class]) / sum(exp(x))))

    weight is a 1D tensor assignning weight to each of the classes.
    '''
    def __init__(self, weight: Tensor = None) -> None:
        if weight is not None:
            assert len(weight.shape) == 1
            self._weight = np.asarray(weight)
        else:
            self._weight = None

    def loss(self, predicted: Tensor, actual: int) -> float:
        m = predicted.shape[0]
        exps = np.exp(predicted - np.max(predicted))
        p = exps / np.sum(exps)
        nll = -np.log(p[range(m), actual])
        if self._weight is not None:
            nll *= self._weight[actual]
        return np.sum(nll)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        m = predicted.shape[0]
        grad = np.copy(predicted)
        grad[range(m), actual] -= 1.0
        return grad
