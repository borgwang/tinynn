"""Model class manage the network, loss function and optimizer."""

import pickle


class Model:

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, predictions, targets):
        loss = self.loss.loss(predictions, targets)
        grad_from_loss = self.loss.grad(predictions, targets)
        struct_grad = self.net.backward(grad_from_loss)
        return loss, struct_grad

    def apply_grads(self, grads):
        params = self.net.params
        self.optimizer.step(grads, params)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.net.params, f)

    def load(self, path):
        with open(path, "rb") as f:
            params = pickle.load(f)

        self.net.params = params
        for layer in self.net.layers:
            layer.is_init = True

    @property
    def is_training(self):
        return self.net.is_training

    @is_training.setter
    def is_training(self, is_training):
        self.net.is_training = is_training
