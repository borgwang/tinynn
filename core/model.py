"""Model class manage the network, loss function and optimizer."""

import pickle


class Model(object):

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

        self._phase = "TRAIN"

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
        for step, (param, _) in zip(steps, self.net.get_params_and_grads()):
            for k, v in param.items():
                param[k] += step[k]

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
        return self._phase

    def set_phase(self, phase):
        assert phase in ("TRAIN", "TEST")
        self.net.set_phase(phase)
        self._phase = phase
