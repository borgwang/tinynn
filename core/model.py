# Author: borgwang <borgwang@126.com>
# Date: 2018-05-23
#
# Filename: model.py
# Description: Model class handles network, loss function and optimizer


import numpy as np

from core.nn import NeuralNet


class Model(object):

    def __init__(self, net, loss_fn, optimizer):
        self.net = net
        self.loss_fn = loss_fn
        self.optim = optimizer

        self.is_training = True

    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, predicted, targets):
        loss = self.loss_fn.loss(predicted, targets)
        grad = self.loss_fn.grad(predicted, targets)
        self.net.backward(grad)
        grads = [grad for param, grad in self.net.get_params_and_grads()]
        return loss, grads

    def apply_grad(self, grads):
        flatten_grads = np.concatenate([np.ravel(grad) for grad in grads])
        flatten_step = self.optim._compute_step(flatten_grads)

        p = 0
        step = []
        for param, grad in self.net.get_params_and_grads():
            block = np.prod(param.shape)
            _step = flatten_step[p: p+block].reshape(param.shape) - \
                self.optim.weight_decay * param
            step.append(_step)
            param += _step
            p += block

        return step
