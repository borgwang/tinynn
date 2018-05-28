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
from core.evaluator import EVEvaluator, MSEEvaluator, MAEEvaluator
from data_processor.data_iterator import BatchIterator


def main():
    # data preparing
    img_path = 'examples/data/origin.jpg'
    if not os.path.isfile(img_path):
        raise FileExistsError('Please put an image name \'origin.jpg\' in %s' % img_path)
    img = np.asarray(Image.open(img_path), dtype='float32') / 255.0

    train_X, train_Y = [], []
    h, w, _ = img.shape
    for r in range(h):
        for c in range(w):
            train_X.append([(r-h/2.0)/h, (c-w/2.0)/w])
            train_Y.append(img[r][c])

    train_X = np.asarray(train_X)
    # train_Y = np.reshape(train_Y, (-1, 1))
    train_Y = np.asarray(train_Y)

    net = NeuralNet([
        Linear(2, 30),
        ReLU(),
        Linear(30, 60),
        ReLU(),
        Linear(60, 60),
        ReLU(),
        Linear(60, 30),
        ReLU(),
        Linear(30, 3),
        Sigmoid()
    ])

    model = Model(net=net, loss_fn=MSELoss(), optimizer=Adam())
    model.initialize()
    ev_evaluator = EVEvaluator()
    mse_evaluator = MSEEvaluator()
    mae_evaluator = MAEEvaluator()
    iterator = BatchIterator(batch_size=32)
    for epoch in range(100):
        t_start = time.time()
        for batch in iterator(train_X, train_Y):
            preds = model.forward(batch.inputs)
            loss, grads = model.backward(preds, batch.targets)
            model.apply_grad(grads)

        # evaluate
        preds = net.forward(train_X)
        ev = ev_evaluator.eval(preds, train_Y)
        mse = mse_evaluator.eval(preds, train_Y)
        mae = mae_evaluator.eval(preds, train_Y)
        print(ev, mse, mae)
        
        # # genrerate painting
        # preds = preds.reshape(h, w, -1)
        # preds = (preds * 255.0).astype('uint8')
        # Image.fromarray(preds).save('examples/data/painting-%d.jpg' % epoch)
        # print('Epoch %d time cost: %.2f' % (epoch, time.time() - t_start))


if __name__ =='__main__':
    main()
