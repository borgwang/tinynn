"""Model class manage the network, loss function and optimizer."""

import copy
import pickle

import numpy as np


class Model(object):

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, preds, targets):
        loss = self.loss.loss(preds, targets)
        grad = self.loss.grad(preds, targets)
        grads, _ = self.net.backward(grad)
        return loss, grads

    @staticmethod
    def _apply_grad(net, optimizer, grads):
        params = net.get_parameters()
        steps = optimizer.compute_step(grads, params)
        for step, param in zip(steps, params):
            for k, v in param.items():
                param[k] += step[k]

    def apply_grad(self, grads):
        self._apply_grad(self.net, self.optimizer, grads)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.net, f, -1)
        print("Model saved in %s." % path)

    def load(self, path):
        # compatibility checking
        with open(path, "rb") as f:
            net = pickle.load(f)
        for l1, l2 in zip(self.net.layers, net.layers):
            if l1.shape != l2.shape:
                raise ValueError("Incompatible architecture. %s in loaded model"
                                 " and %s in defined model." %
                                 (l1.shape, l2.shape))
            else:
                print("%s: %s" % (l1.name, l1.shape))
        self.net = net
        print("Restored model from %s." % path)

    def get_phase(self):
        return self.net.get_phase()

    def set_phase(self, phase):
        self.net.set_phase(phase)


class AutoEncoder(Model):
    def __init__(self, encoder, decoder, loss, optimizer):
        super().__init__(net=None, loss=loss, optimizer=None)
        self.encoder = encoder
        self.decoder = decoder
        self.en_opt = optimizer
        self.de_opt = copy.deepcopy(optimizer)

    def forward(self, inputs):
        code = self.encoder.forward(inputs)
        genn = self.decoder.forward(code)
        return genn

    def backward(self, preds, targets):
        # calculate loss
        loss = self.loss.loss(preds, targets)
        grad = self.loss.grad(preds, targets)
        # backprop gradients through decoder and encoder
        de_grads, de_grad = self.decoder.backward(grad)
        en_grads, _ = self.encoder.backward(de_grad)
        return loss, np.array([en_grads, de_grads])

    def apply_grad(self, grads):
        en_grads, de_grads = (grads[0], grads[1])
        self._apply_grad(self.decoder, self.de_opt, de_grads)
        self._apply_grad(self.encoder, self.en_opt, en_grads)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.encoder, self.decoder), f, -1)

    def load(self, path):
        with open(path, "rb") as f:
            encoder, decoder = pickle.load(f)
        self.encoder = encoder
        self.decoder = decoder

    def get_phase(self):
        return self.encoder.get_phase()

    def set_phase(self, phase):
        self.encoder.set_phase(phase)
        self.decoder.set_phase(phase)
