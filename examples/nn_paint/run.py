# Author: borgwang <borgwang@126.com>
# Date: 2018-05-20
#
# Filename: run.py
# Description: Use a neural network to mimic images.

import sys
import os
sys.path.append(os.getcwd())

import time
import numpy as np
from PIL import Image

from core.nn import NeuralNet
from core.layers import Linear, ReLU, Sigmoid
from core.optimizer import Momentum, Adam
from core.loss import MSELoss
from core.model import Model
from data_processor.data_iterator import BatchIterator


def main():
    # data preparing
    img = np.asarray(Image.open('examples/data/origin.jpg'), dtype='float32')
    img /= 255.0

    train_X, train_Y = [], []
    h, w, _ = img.shape
    for r in range(h):
        for c in range(w):
            train_X.append([(r-h/2.0)/h, (c-w/2.0)/w])
            train_Y.append(img[r][c])

    train_X = np.asarray(train_X)
    train_Y = np.asarray(train_Y)

    net = NeuralNet([
        Linear(num_in=2, num_out=30),
        ReLU(),
        Linear(num_in=30, num_out=60),
        ReLU(),
        Linear(num_in=60, num_out=60),
        ReLU(),
        Linear(num_in=60, num_out=30),
        ReLU(),
        Linear(num_in=30, num_out=3),
        Sigmoid()
    ])

    iterator = BatchIterator(batch_size=32)
    model = Model(net=net, loss_fn=MSELoss(), optimizer=Adam())

    for epoch in range(100):
        t_start = time.time()
        for batch in iterator(train_X, train_Y):
            preds = model.forward(batch.inputs)
            loss, grads = model.backward(preds, batch.targets)
            model.apply_grad(grads)
        # genrerate painting
        preds = net.forward(train_X)
        preds = preds.reshape(h, w, -1)
        preds = (preds * 255.0).astype('uint8')
        Image.fromarray(preds).save('examples/data/painting-%d.jpg' % epoch)
        print('Epoch %d time cost: %.2f' % (epoch, time.time() - t_start))


if __name__ =='__main__':
    main()
