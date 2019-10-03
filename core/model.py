"""Model class manage the network, loss function and optimizer."""

import pickle


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
        grad_struct = self.net.backward(grad)
        return loss, grad_struct

    def apply_grad(self, grads):
        params = self.net.params
        self.optimizer.step(grads, params)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.net, f, -1)

    def load(self, path):
        with open(path, "rb") as f:
            net = pickle.load(f)
        self.net = net

    def get_phase(self):
        return self.net.get_phase()

    def set_phase(self, phase):
        self.net.set_phase(phase)
