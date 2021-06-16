"""Feed-forward Neural Network class."""

import copy

import numpy as np

from tinynn.utils.structured_param import StructuredParam


class Net:

    def __init__(self, layers):
        self.layers = layers
        self._phase = "TRAIN"

    def __repr__(self):
        return "\n".join([str(l) for l in self.layers])

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
        trainable = [l.params for l in self.layers]
        untrainable = [l.ut_params for l in self.layers]
        return StructuredParam(trainable, untrainable)

    @params.setter
    def params(self, params):
        self.params.values = params.values
        self.params.ut_values = params.ut_values

    def get_phase(self):
        return self._phase

    def set_phase(self, phase):
        for layer in self.layers:
            layer.set_phase(phase)
        self._phase = phase

    def init_params(self, input_shape):
        # manually init params by letting data forward through the network
        self.forward(np.ones((1, *input_shape)))

