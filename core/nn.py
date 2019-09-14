"""Feed-forward Neural Network class."""


class Net(object):

    def __init__(self, layers):
        self.layers = layers
        self._phase = "TRAIN"

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        all_grads = []
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            all_grads.append(layer.grads)
        return (all_grads[::-1], grad)

    def get_parameters(self):
        return [layer.params for layer in self.layers]

    def set_parameters(self, params):
        for i, layer in enumerate(self.layers):
            assert layer.params.keys() == params[i].keys()
            for key in layer.params.keys():
                assert layer.params[key].shape == params[i][key].shape
                layer.params[key] = params[i][key]

    def get_phase(self):
        return self._phase

    def set_phase(self, phase):
        for layer in self.layers:
            layer.set_phase(phase)
        self._phase = phase
