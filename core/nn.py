"""Feed-forward Neural Network class."""

import copy

import numpy as np


class Net(object):

    def __init__(self, layers):
        self.layers = layers
        self._phase = "TRAIN"

        self._param_struct = self.load_parameters()
        
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        grad_strt = ParamStruct()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_strt.layer_data.insert(0, layer.grads)
        grad_strt.wrt_input = grad
        return grad_strt

    def load_parameters(self):
        param_struct = ParamStruct()
        for layer in self.layers:
            param_struct.layer_data.append(layer.params)
        return param_struct

    @property
    def params(self):
        return self._param_struct

    @params.setter
    def params(self, params):
        self._param_struct.from_values(params.values)

    def get_phase(self):
        return self._phase

    def set_phase(self, phase):
        for layer in self.layers:
            layer.set_phase(phase)
        self._phase = phase


class ParamStruct(object):
    """
    A helper class represents network parameters or gradients.
    TODO: a better class name
    """

    def __init__(self):
        self.layer_data = []

    @property
    def values(self):
        values = list()
        for d in self.layer_data:
            for v in d.values():
                values.append(v)
        return np.array(values)

    def from_values(self, values):
        i = 0
        for d in self.layer_data:
            for name in d.keys():
                d[name] = values[i]
                i += 1

    @staticmethod
    def _ensure_values(obj):
        if isinstance(obj, ParamStruct):
            obj = obj.values
        return obj

    def __add__(self, other):
        other = self._ensure_values(other)
        obj = copy.deepcopy(self)
        obj.from_values(self.values + other)
        return obj

    def __radd__(self, other):
        other = self._ensure_values(other)
        obj = copy.deepcopy(self)
        obj.from_values(other + self.values)
        return obj

    def __iadd__(self, other):
        other = self._ensure_values(other)
        self.from_values(self.values + other)
        return self

    def __sub__(self, other):
        other = self._ensure_values(other)
        obj = copy.deepcopy(self)
        obj.from_values(self.values - other)
        return obj

    def __rsub__(self, other):
        other = self._ensure_values(other)
        obj = copy.deepcopy(self)
        obj.from_values(other - self.values)
        return obj

    def __isub__(self, other):
        other = self._ensure_values(other)
        self.from_values(self.values - other)
        return self

    def __mul__(self, other):
        other = self._ensure_values(other)
        obj = copy.deepcopy(self)
        obj.from_values(self.values * other)
        return obj

    def __rmul__(self, other):
        other = self._ensure_values(other)
        obj = copy.deepcopy(self)
        obj.from_values(other * self.values)
        return obj

    def __imul__(self, other):
        other = self._ensure_values(other)
        self.from_values(self.values * other)
        return self

    def __neg__(self):
        obj = copy.deepcopy(self)
        obj.from_values(-self.values)
        return obj
