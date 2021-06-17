"""Feed-forward Neural Network class."""

import copy

import numpy as np

from tinynn.utils.structured_param import StructuredParam


class Net:

    def __init__(self, layers):
        self.layers = layers
        self._is_training = True

    def __repr__(self):
        return "\n".join([str(layer) for layer in self.layers])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        # back propagation
        layer_grads = []
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            layer_grads.append(copy.copy(layer.grads))

        # return structured gradients
        struct_grad = StructuredParam(layer_grads[::-1])
        # keep the gradient w.r.t the input
        struct_grad.wrt_input = grad
        return struct_grad

    @property
    def params(self):
        trainable = [layer.params for layer in self.layers]
        non_trainable = [layer.nt_params for layer in self.layers]
        return StructuredParam(trainable, non_trainable)

    @params.setter
    def params(self, params):
        self.params.values = params.values
        self.params.nt_values = params.nt_values

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, is_training):
        for layer in self.layers:
            layer.is_training = is_training
        self._is_training = is_training

    def init_params(self, input_shape):
        # manually init params by letting data forward through the network
        self.forward(np.ones((1, *input_shape)))
