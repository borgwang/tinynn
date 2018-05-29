# Author: borgwang <borgwang@126.com>
# Date: 2018-05-05
#
# Filename: nn.py
# Description: Feedforwad Neural Network class.


import numpy as np

from core.layers import Dropout


class NeuralNet(object):

    def __init__(self, layers):
        self.layers = layers
        # TODO: better way to handle train/test phase
        self.training = True

    def forward(self, inputs):
        for layer in self.layers:
            # TODO: Turn off dropout at test time
            if isinstance(layer, Dropout) and not self.training:
                continue
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        all_grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, Dropout) and not self.training:
                continue
            grad = layer.backward(grad)
            all_grads.append(layer.grads)
        return all_grads[::-1]

    def initialize(self):
        for layer in self.layers:
            layer.initializate()

    def get_params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def set_training_phase(self):
        self.training = True

    def set_test_phase(self):
        self.training = False

    def get_parameters(self):
        return [layer.params for layer in self.layers]

    def set_parameters(self, params):
        for i, layer in enumerate(self.layers):
            assert layer.params.keys() == params[i].keys()
            for key in layer.params.keys():
                assert layer.params[key].shape == params[i][key].shape
                layer.params[key] = params[i][key]
