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
        self.training = True
        
    def forward(self, inputs):
        for layer in self.layers:
            # TODO: Turn off dropout at test time
            if isinstance(layer, Dropout) and not self.training:
                continue
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            if isinstance(layer, Dropout) and not self.training:
                continue
            grad = layer.backward(grad)
        return grad

    def get_params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def set_training_phase(self):
        self.training = True

    def set_test_phase(self):
        self.training = False
