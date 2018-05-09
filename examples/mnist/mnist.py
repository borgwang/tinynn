import sys
import os
sys.path.append(os.getcwd())
import time
import numpy as np

from core.train import train, evaluate
from core.nn import NeuralNet
from core.layers import Linear, Tanh, ReLU, Sigmoid
from core.optimizer import SGD, Adam, RMSProp, Momentum, StepLR, MultiStepLR, LinearLR, ExponentialLR
from core.loss import CrossEntropyLoss
from core.initializer import NormalInit, TruncatedNormalInit, UniformInit, ZerosInit, ConstantInit, XavierUniformInit, XavierNormalInit, OrthogonalInit

from core.data.data import DataIterator, BatchIterator
from core.data.dataset import MNIST

import matplotlib.pyplot as plt

# ZOO = {
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
train_Y = get_one_hot(train_Y, 10)
valid_Y = get_one_hot(valid_Y, 10)

# train_X = np.concatenate([train_X, valid_X])
# train_Y = np.concatenate([train_Y, valid_Y])
# train_Y = get_one_hot(np.concatenate([train_Y, valid_Y]), 10)

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
optimizer = Adam(1e-3)
# lr_scheduler = MultiStepLR(optimizer, milestones=[1, 3, 5], gamma=0.5)
# lr_scheduler = LinearLR(optimizer, decay_steps=25, final_lr=1e-4)
lr_scheduler = ExponentialLR(optimizer, decay_steps=5)

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
    print('lr:', lr_scheduler.step())
    print('Epoch %d timecost: %.4f' % (epoch, time.time() - t_start))
    valid_acc = evaluate(net, valid_X, valid_Y)
    epoch_accs.append(valid_acc)

import pdb; pdb.set_trace()
