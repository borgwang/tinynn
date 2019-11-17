"""Feed-forward Neural Network class."""

import copy

import numpy as np


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


class StructuredParam:
    """A helper class represents network parameters or gradients."""

    def __init__(self, param_list, ut_param_list=None):
        self.param_list = param_list
        self.ut_param_list = ut_param_list

    @property
    def values(self):
        return np.array([v for p in self.param_list for v in p.values()])

    @values.setter
    def values(self, values):
        i = 0
        for d in self.param_list:
            for name in d.keys():
                d[name] = values[i]
                i += 1

    @property
    def ut_values(self):
        return np.array([v for p in self.ut_param_list for v in p.values()])

    @ut_values.setter
    def ut_values(self, values):
        i = 0
        for d in self.ut_param_list:
            for name in d.keys():
                d[name] = values[i]
                i += 1

    @property
    def shape(self):
        shape = list()
        for d in self.param_list:
            l_shape = dict()
            for k, v in d.items():
                l_shape[k] = v.shape
            shape.append(l_shape)
        shape = tuple(shape)
        return shape

    @staticmethod
    def _ensure_values(obj):
        if isinstance(obj, StructuredParam):
            obj = obj.values
        return obj

    def clip(self, min_=None, max_=None):
        obj = copy.deepcopy(self)
        obj.values = [v.clip(min_, max_)for v in self.values]
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
        obj.values = self.values * self._ensure_values(other)
        return obj

    def __rmul__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) * self.values
        return obj

    def __imul__(self, other):
        self.values *= self._ensure_values(other)
        return self

    def __truediv__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self.values / self._ensure_values(other)
        return obj

    def __rtruediv__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) / self.values
        return obj

    def __itruediv__(self, other):
        self.values /= self._ensure_values(other)
        return self

    def __pow__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self.values ** self._ensure_values(other)
        return obj

    def __ipow__(self, other):
        self.values **= self._ensure_values(other)
        return self

    def __neg__(self):
        obj = copy.deepcopy(self)
        obj.values = -self.values
        return obj

    def __len__(self):
        return len(self.values)

    def __lt__(self, other):
        obj = copy.deepcopy(self)
        other = self._ensure_values(other)

        if isinstance(other, float):
            obj.values = [v < other for v in self.values]
        else:
            obj.values = [v < other[i] for i, v in enumerate(self.values)]
        return obj

    def __gt__(self, other):
        obj = copy.deepcopy(self)
        other = self._ensure_values(other)

        if isinstance(other, float):
            obj.values = [v > other for v in self.values]
        else:
            obj.values = [v > other[i] for i, v in enumerate(self.values)]
        return obj

    def __and__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) & self.values
        return obj

    def __or__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) | self.values
        return obj
