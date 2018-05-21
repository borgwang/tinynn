# Author: borgwang <borgwang@126.com>
# Date: 2018-05-08
#
# Filename: run.py
# Description: Example MNIST code for tinynn.


import sys
import os
sys.path.append(os.getcwd())
import time
import numpy as np

from core.train import train, evaluate
from core.nn import NeuralNet
from core.layers import Linear, Tanh, ReLU, Sigmoid, LeakyReLU, Dropout
from core.optimizer import SGD, Adam, RMSProp, Momentum, StepLR, MultiStepLR, LinearLR, ExponentialLR
from core.loss import CrossEntropyLoss
from core.initializer import NormalInit, TruncatedNormalInit, UniformInit, ZerosInit, ConstantInit, XavierUniformInit, XavierNormalInit, OrthogonalInit

from core.data.data import DataIterator, BatchIterator
from core.data.dataset import MNIST

import matplotlib.pyplot as plt


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

mnist = MNIST('./examples/data', transform=None)
train_X, train_Y = mnist.get_train_data()
valid_X, valid_Y = mnist.get_valid_data()
test_X, test_Y = mnist.get_test_data()
train_Y = get_one_hot(train_Y, 10)
valid_Y = get_one_hot(valid_Y, 10)
valid_Y = get_one_hot(valid_Y, 10)

# train_X = np.concatenate([train_X, valid_X])
# train_Y = np.concatenate([train_Y, valid_Y])
# train_Y = get_one_hot(train_Y, 10)
# test_Y = get_one_hot(test_Y, 10)

net = NeuralNet([
    Linear(num_in=784, num_out=200),
    ReLU(),
    Linear(num_in=200, num_out=50),
    ReLU(),
    Linear(num_in=50, num_out=10)
])

num_epochs = 50
iterator = BatchIterator(batch_size=32)
loss = CrossEntropyLoss()
optimizer = Adam(lr=1e-3, weight_decay=0.001)
# lr_scheduler = ExponentialLR(optimizer, decay_steps=50)

batch_losses = []
epoch_accs = []
for epoch in range(num_epochs):
    t_start = time.time()
    for batch in iterator(train_X, train_Y):
        predicted = net.forward(batch.inputs)
        loss_ = loss.loss(predicted, batch.targets)
        batch_losses.append(loss_)
        grad = loss.grad(predicted, batch.targets)
        net.backward(grad)
        optimizer.step(net)
    # print('lr:', lr_scheduler.step())
    print('Epoch %d timecost: %.4f' % (epoch, time.time() - t_start))
    test_acc = evaluate(net, test_X, test_Y)
    epoch_accs.append(test_acc)
