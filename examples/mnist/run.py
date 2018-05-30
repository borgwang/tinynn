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
import matplotlib.pyplot as plt

from core.nn import NeuralNet
from core.layers import Linear, Conv2D, Flatten, Tanh, ReLU, Sigmoid, LeakyReLU, Dropout
from core.optimizer import SGD, Adam, RMSProp, Momentum, StepLR, MultiStepLR, LinearLR, ExponentialLR
from core.loss import CrossEntropyLoss
from core.initializer import NormalInit, TruncatedNormalInit, UniformInit, ZerosInit, ConstantInit, XavierUniformInit, XavierNormalInit, OrthogonalInit
from core.model import Model
from core.evaluator import AccEvaluator
from utils.seeder import random_seed

from data_processor.dataset import MNIST
from data_processor.data_iterator import BatchIterator


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

    random_seed(args.seed)

    # build model
    # net = NeuralNet([
    #     Linear(784, 200),
    #     ReLU(),
    #     # Dropout(),
    #     Linear(200, 100),
    #     ReLU(),
    #     Linear(100, 70),
    #     ReLU(),
    #     Linear(70, 30),
    #     ReLU(),
    #     Linear(30, 10)
    # ])
    c1 = Conv2D((28, 28, 1), channels=4, kernel_size=5, stride=1)
    c1_a = ReLU()
    c2 = Conv2D(conv1.out_dim, channels=8, kernel_size=5, stride=2)
    c2_a = ReLU()
    c3 = Conv2D(conv2.out_dim, channels=12, kernel_size=4, stride=2)
    c3_a = ReLU()
    flat = Flatten(conv3.out_dim)
    f1 = Linear(flat.out_dim, 70)
    f1_a = ReLU()
    f2 = Linear(70, 10)

    net = NeuralNet([c1, c1_a, c2, c2_a, c3, c3_a, flat, f1, f1_a, f2])
    

    import pdb; pdb.set_trace()
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
    model.initialize()
    # model.load('examples/data/model.pk')

    iterator = BatchIterator(batch_size=args.batch_size)
    evaluator = AccEvaluator()
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_X, train_Y):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grad(grads)
        # print('current lr: %.4f' % lr_scheduler.step())
        print('Epoch %d timecost: %.4f' % (epoch, time.time() - t_start))
        # evaluate
        model.set_phase('TEST')
        test_pred = model.forward(test_X)
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_Y_idx = np.asarray(test_Y)
        res = evaluator.eval(test_pred_idx, test_Y_idx)
        print(res)
        model.set_phase('TRAIN')
    # model.save('examples/data/model.pk')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ep', default=100, type=int)
    parser.add_argument('--data_path', default='./examples/data', type=str)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    main(args)
