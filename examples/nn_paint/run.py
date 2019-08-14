# Author: borgwang <borgwang@126.com>
# Date: 2018-05-20
#
# Filename: run.py
# Description: Use a neural network to mimic images.

import runtime_path  # isort:skip

import argparse
import os
import time

import numpy as np
from PIL import Image

from core.evaluator import EVEvaluator
from core.evaluator import MSEEvaluator
from core.layers import Linear
from core.layers import ReLU
from core.layers import Sigmoid
from core.loss import MSELoss
from core.model import Model
from core.nn import NeuralNet
from core.optimizer import Adam
from utils.data_iterator import BatchIterator


def prepare_dataset(img_path):
    if not os.path.isfile(img_path):
        raise FileExistsError("Image %s not exist" % img_path)
    img = np.asarray(Image.open(img_path), dtype="float32") / 255.0

    train_x, train_y = [], []
    h, w, _ = img.shape
    for r in range(h):
        for c in range(w):
            train_x.append([(r - h / 2.0) / h, (c - w / 2.0) / w])
            train_y.append(img[r][c])
    return np.asarray(train_x), np.asarray(train_y)


def main(args):
    # data preparing
    data_path = os.path.join(args.data_dir, args.file_name)
    train_x, train_y = prepare_dataset(data_path)

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

    model = Model(net=net, loss=MSELoss(), optimizer=Adam())
    model.initialize()
    ev_evaluator = EVEvaluator()
    mse_evaluator = MSEEvaluator()
    iterator = BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_x, train_y):
            preds = model.forward(batch.inputs)
            loss, grads = model.backward(preds, batch.targets)
            model.apply_grad(grads)

        # evaluate
        preds = net.forward(train_x)
        ev = ev_evaluator.evaluate(preds, train_y)
        mse = mse_evaluator.evaluate(preds, train_y)
        print(ev, mse)

        if args.paint:
            # generate painting
            preds = preds.reshape(h, w, -1)
            preds = (preds * 255.0).astype("uint8")
            filename, ext = os.path.splitext(args.file_name)
            output_filename = "output" + ext
            output_path = os.path.join(args.data_dir, output_filename)
            Image.fromarray(preds).save(output_path)
        print("Epoch %d time cost: %.2f" % (epoch, time.time() - t_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./examples/nn_paint/data", type=str)
    parser.add_argument("--file_name", default="input.jpg", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_ep", default=100, type=int)
    parser.add_argument("--paint", default=True, type=bool)
    main(parser.parse_args())
