# Author: borgwang <borgwang@126.com>
# Date: 2018-05-08
#
# Filename: run.py
# Description: Example MNIST code for tinynn.


import sys
import os
sys.path.append(os.getcwd())
import time
import argparse
import numpy as np

from core.nn import NeuralNet
from core.layers import Linear, Tanh, ReLU, Sigmoid, LeakyReLU, Dropout
from core.optimizer import SGD, Adam, RMSProp, Momentum, StepLR, MultiStepLR, LinearLR, ExponentialLR
from core.loss import CrossEntropyLoss
from core.initializer import NormalInit, TruncatedNormalInit, UniformInit, ZerosInit, ConstantInit, XavierUniformInit, XavierNormalInit, OrthogonalInit
from core.model import Model

from data_processor.dataset import MNIST
from data_processor.data_iterator import BatchIterator

import matplotlib.pyplot as plt


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def main(args):
    # data preparing
    mnist = MNIST(args.data_path, transform=None)
    train_X, train_Y = mnist.train_data
    valid_X, valid_Y = mnist.valid_data
    test_X, test_Y = mnist.test_data
    train_Y = get_one_hot(train_Y, 10)
    valid_Y = get_one_hot(valid_Y, 10)

    # build model
    net = NeuralNet([
        Linear(num_in=784, num_out=200),
        ReLU(),
        Linear(num_in=200, num_out=100),
        ReLU(),
        Linear(num_in=100, num_out=50),
        ReLU(),
        Linear(num_in=50, num_out=10)
    ])
    loss_fn = CrossEntropyLoss()

    if args.optim == 'adam':
        optimizer = Adam(lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = SGD(lr=args.lr)
    elif args.optim == 'momentum':
        optimizer = Momentum(lr=args.lr)
    elif args.optim == 'rmsprop':
        optimizer = RMSProp(lr=args.lr)
    else:
        raise ValueError('Invalid Optimizer!!')

    # lr_scheduler = ExponentialLR(optimizer, decay_steps=50)
    model = Model(net=net, loss_fn=loss_fn, optimizer=optimizer)

    # train
    iterator = BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_X, train_Y):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grad(grads)
        # print('current lr: %.4f' % lr_scheduler.step())
        print('Epoch %d timecost: %.4f' % (epoch, time.time() - t_start))

        # evaluate
        model.is_training = False
        test_pred = model.forward(test_X)
        test_pred_idx = np.argmax(test_pred, axis=1)

        if len(test_Y.shape) == 1:
            assert(len(test_Y) == len(test_pred_idx))
            test_Y_idx = np.asarray(test_Y)
        elif len(test_Y.shape) == 2:
            test_Y_idx = np.argmax(test_Y, axis=1)
        else:
            raise ValueError('Target Tensor dimensional error!')

        accuracy = np.sum(test_pred_idx == test_Y_idx) / len(test_Y)
        print('Accuracy on %d data: %.2f%%' % (len(test_Y), accuracy * 100))
        model.is_training = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ep', default=100, type=int)
    parser.add_argument('--data_path', default='./examples/data', type=str)
    parser.add_argument('--optim', default='adam', type=str, help='[adam|sgd|momentum|rmsprop]')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    main(args)
