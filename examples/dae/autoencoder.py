"""AutoEncoder class."""

from tinynn.core.model import Model


class AutoEncoder(Model):

    def __init__(self, net, loss, optimizer):
        super().__init__(net, loss, optimizer)
        self.en_net = net[0]
        self.de_net = net[1]

    def forward(self, inputs):
        for net in self.net:
            inputs = net.forward(inputs)
        return inputs

    def backward(self, preds, targets):
        loss = self.loss.loss(preds, targets)
        grad_from_loss = self.loss.grad(preds, targets)

        # through decoder network
        de_grads = self.de_net.backward(grad_from_loss)
        # back-propagate to encoder network
        grads_to_en_net = de_grads.wrt_input
        en_grads = self.en_net.backward(grads_to_en_net)

        return loss, (en_grads, de_grads)

    def apply_grads(self, grads):
        for net, grad, opt in zip(self.net, grads, self.optimizer):
            opt.step(grad, net.params)
