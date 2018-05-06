import numpy as np
from typing import Sequence, Iterator, Tuple

from core.tensor import Tensor
from core.layers import Layer


class NeuralNet(object):

    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
        self.num_params = sum([sum([np.prod(p.shape) for p in layer.params.values()]) for layer in layers])

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def get_params(self) -> Iterator[Tensor]:
        for layer in self.layers:
            for name, param in layer.params.items():
                yield param
