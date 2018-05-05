from core.nn import NeuralNet


class Optimizer(object):

    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
