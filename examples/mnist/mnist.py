import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from core.train import train, evaluate
from core.nn import NeuralNet
from core.layers import Linear, Tanh, ReLU, Sigmoid
from core.optimizer import SGD, Adam, RMSProp, Momentum
from core.loss import CrossEntropyLoss
from core.initializer import NormalInit, UniformInit, ZerosInit, ConstantInit, XavierUniformInit, XavierNormalInit, OrthogonalInit

from core.data.data import DataIterator, BatchIterator
from core.data.dataset import MNIST

import matplotlib.pyplot as plt

# MODEL_ZOO = {
#     'linear': [
#         Linear(num_in=784, num_out=10)
#     ],
#     '5-layers-sigmoid': [
#         Linear(num_in=784, num_out=200),
#         Sigmoid(),
#         Linear(num_in=200, num_out=100),
#         Sigmoid(),
#         Linear(num_in=100, num_out=60),
#         Sigmoid(),
#         Linear(num_in=60, num_out=30),
#         Sigmoid(),
#         Linear(num_in=30, num_out=10)
#     ]
# }

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


mnist = MNIST('./examples/data', transform=None)
train_X, train_Y = mnist.get_train_data()
valid_X, valid_Y = mnist.get_valid_data()

# train_X = np.concatenate([train_X, valid_X])
# train_Y = np.concatenate([train_Y, valid_Y])
# train_Y = get_one_hot(np.concatenate([train_Y, valid_Y]), 10)


net = NeuralNet([
    Linear(num_in=784, num_out=100, w_init=XavierNormalInit()),
    Sigmoid(),
    Linear(num_in=100, num_out=10, w_init=XavierNormalInit())
])

num_epochs = 50
eval_interval = 10
iterator = BatchIterator(batch_size=32)
loss = CrossEntropyLoss()
optimizer = Adam(3e-4)

epoch_loss = []
for epoch in range(num_epochs):
    for batch in iterator(train_X, train_Y):
        predicted = net.forward(batch.inputs)
        loss_ = loss.loss(predicted, batch.targets)
        epoch_loss.append(loss_)
        grad = loss.grad(predicted, batch.targets)
        net.backward(grad)
        optimizer.step(net)
    print(epoch, np.mean(epoch_loss))
    if epoch % eval_interval == 0:
        evaluate(net, valid_X, valid_Y)
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.show()
