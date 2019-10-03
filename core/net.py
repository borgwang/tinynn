"""Feed-forward Neural Network class."""

import copy

import numpy as np


class Net(object):

    def __init__(self, layers):
        self.layers = layers
        self._phase = "TRAIN"

        layer_params = [l.params for l in self.layers]
        self._struct_param = StructuredParam(layer_params)

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
        return self._struct_param

    @params.setter
    def params(self, params):
        self._struct_param.values = params.values

    def get_phase(self):
        return self._phase

    def set_phase(self, phase):
        for layer in self.layers:
            layer.set_phase(phase)
        self._phase = phase


class StructuredParam(object):
    """A helper class represents network parameters or gradients."""

    def __init__(self, layer_data):
        self.layer_data = layer_data
        self._values = None

    def _get_values(self):
        values = list()
        for d in self.layer_data:
            for v in d.values():
                values.append(v)
        return np.array(values)

    @property
    def values(self):
        if self._values is None:
            self._values = self._get_values()
        return self._values

    @values.setter
    def values(self, values):
        i = 0
        for d in self.layer_data:
            for name in d.keys():
                d[name] = values[i]
                i += 1
        self._values = self._get_values()

    @staticmethod
    def _ensure_values(obj):
        if isinstance(obj, StructuredParam):
            obj = obj.values
        return obj

    def __add__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self.values + self._ensure_values(other)
        return obj

    def __radd__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) + self.values
        return obj

    def __iadd__(self, other):
        self.values += self._ensure_values(other)
        return self

    def __sub__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self.values - self._ensure_values(other)
        return obj

    def __rsub__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) - self.values
        return obj

    def __isub__(self, other):
        other = self._ensure_values(other)
        self.values -= self._ensure_values(other)
        return self

    def __mul__(self, other):
        obj = copy.deepcopy(self)
        obj.valus = self.values * self._ensure_values(other)
        return obj

    def __rmul__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) * self.values
        return obj

    def __imul__(self, other):
        self.values *= self._ensure_values(other)
        return self

    def __neg__(self):
        obj = copy.deepcopy(self)
        obj.values = -self.values
        return obj
