import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from core.train import train, evaluate
from core.nn import NeuralNet
from core.layers import Linear, Tanh
from core.optimizer import SGD, Adam, RMSProp, Momentum
from core.loss import CrossEntropyLoss
from core.initializer import NormalInit, UniformInit, ZerosInit, ConstantInit, XavierUniformInit, XavierNormalInit, OrthogonalInit, SparseInit

from core.data.data import DataIterator, BatchIterator
from core.data.dataset import MNIST


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

mnist = MNIST('./examples/data', transform=None)
train_X, train_Y = mnist.get_train_data()
valid_X, valid_Y = mnist.get_valid_data()

train_X = np.concatenate([train_X, valid_X])
train_Y = np.concatenate([train_Y, valid_Y])
# train_Y = get_one_hot(np.concatenate([train_Y, valid_Y]), 10)

net = NeuralNet([
    Linear(num_in=784, num_out=100),
    Tanh(),
    Linear(num_in=100, num_out=10)
])

train(net,
      train_X,
      train_Y,
      num_epochs=5000,
      iterator=BatchIterator(batch_size=16),
      loss=CrossEntropyLoss(),
      optimizer=Adam(lr=3e-3))
