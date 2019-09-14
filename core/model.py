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
        grads = self.net.backward(grad)
        return loss, grads

    def apply_grad(self, grads):
        params = self.net.get_parameters()
        steps = self.optimizer.compute_step(grads, params)
        for step, param in zip(steps, params):
            for k, v in param.items():
                param[k] += step[k]

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
