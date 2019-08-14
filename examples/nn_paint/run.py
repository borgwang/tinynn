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
import argparse
from PIL import Image

from core.nn import NeuralNet
from core.layers import Linear, ReLU, Sigmoid
from core.optimizer import Adam
from core.loss import MSELoss
from core.model import Model
from core.evaluator import EVEvaluator, MSEEvaluator
from utils.data_iterator import BatchIterator


def main(args):
    # data preparing
    filename = "origin.jpg"
    img_path = os.path.join(args.dir, filename)
    if not os.path.isfile(img_path):
        raise FileExistsError("Image \"origin.jpg\" not exist in %s" % args.dir)
    img = np.asarray(Image.open(img_path), dtype="float32") / 255.0

    train_X, train_Y = [], []
    h, w, _ = img.shape
    for r in range(h):
        for c in range(w):
            train_X.append([(r-h/2.0)/h, (c-w/2.0)/w])
            train_Y.append(img[r][c])

    train_X = np.asarray(train_X)
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
    iterator = BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_X, train_Y):
            preds = model.forward(batch.inputs)
            loss, grads = model.backward(preds, batch.targets)
            model.apply_grad(grads)

        # evaluate
        preds = net.forward(train_X)
        ev = ev_evaluator.evaluate(preds, train_Y)
        mse = mse_evaluator.evaluate(preds, train_Y)
        print(ev, mse)

        if args.paint:
            # generate painting
            preds = preds.reshape(h, w, -1)
            preds = (preds * 255.0).astype("uint8")
            output_filename = "painting.jpg"
            Image.fromarray(preds).save(os.path.join(args.dir, output_filename))

        print("Epoch %d time cost: %.2f" % (epoch, time.time() - t_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./examples/data", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_ep", default=100, type=int)
    parser.add_argument("--paint", default=True, type=bool)
    args = parser.parse_args()
    main(args)
