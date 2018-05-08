import numpy as np

from core.tensor import Tensor
from core.nn import NeuralNet
from core.loss import Loss, MSELoss
from core.optimizer import Optimizer, Adam
from core.data.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSELoss(),
          optimizer: Optimizer = Adam(3e-4)) -> None:
    for epoch in range(num_epochs):
        epoch_loss = []
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss.append(loss.loss(predicted, batch.targets))
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, np.sum(epoch_loss))


def evaluate(net: NeuralNet,
             inputs: Tensor,
             targets: Tensor) -> None:

    predicted = net.forward(inputs)
    predicted_idx = np.argmax(predicted, axis=1)

    if len(targets.shape) == 1:
        assert(len(targets) == len(predicted_idx))
        target_idx = np.asarray(targets)
    elif len(targets.shape) == 2:
        target_idx = np.argmax(targets, axis=1)
    else:
        raise ValueError('Target Tensor dimensional error!')

    accuracy = np.sum(predicted_idx == target_idx) / len(targets)

    print('Accuracy on %d test data: %.2f%%' % (len(targets), accuracy * 100))
