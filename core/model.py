# Author: borgwang <borgwang@126.com>
# Date: 2018-05-23
#
# Filename: model.py
# Description: Model class handles network, loss function and optimizer


import numpy as np
import pickle

from core.nn import NeuralNet
from utils.timer import Timer


class Model(object):

    def __init__(self, net, loss_fn, optimizer):
        self.net = net
        self.loss_fn = loss_fn
        self.optim = optimizer

        self.is_training = True

    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, preds, targets):
        loss = self.loss_fn.loss(preds, targets)
        grad = self.loss_fn.grad(preds, targets)
        self.net.backward(grad)
        # flatten gradient list in order to compute actual gradient step.
        flatten_grads = np.concatenate(
            [np.ravel(grad) for param, grad in self.net.get_params_and_grads()])
        flatten_step = self.optim._compute_step(flatten_grads)

        step = []
        p = 0
        for param, grad in self.net.get_params_and_grads():
            block = np.prod(param.shape)
            # TODO: Doing weight decay inside optimizer
            _step = flatten_step[p: p+block].reshape(param.shape) - \
                self.optim.weight_decay * param
            step.append(_step)
            p += block
        return loss, step

    def apply_grad(self, grads):
        for grad, (param, _) in zip(grads, self.net.get_params_and_grads()):
            param += grad

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.net, f, -1)
        # TODO: logging module
        print('Model saved in %s.' % path)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.net = pickle.load(f)
        print('Restored model from %s.' % path)
