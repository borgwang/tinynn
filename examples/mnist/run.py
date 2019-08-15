# Author: borgwang <borgwang@126.com>
# Date: 2018-05-08
#
# Filename: run.py
# Description: Example MNIST code for tinynn.

import runtime_path  # isort:skip

import argparse
import gzip
import os
import pickle
import time
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np

from core.evaluator import AccEvaluator
from core.layers import Dense
from core.layers import Conv2D
from core.layers import Flatten
from core.layers import ReLU
from core.layers import Dropout
from core.losses import CrossEntropyLoss
from core.model import Model
from core.nn import Net
from core.optimizer import SGD
from core.optimizer import Adam
from core.optimizer import Momentum
from core.optimizer import RMSProp
from utils.data_iterator import BatchIterator
from utils.seeder import random_seed


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def prepare_dataset(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    path = os.path.join(data_dir, url.split("/")[-1])

    # download
    try:
        if os.path.exists(path):
            print("{} already exists.".format(path))
        else:
            print("Downloading {}.".format(url))
            try:
                urlretrieve(url, path)
            except URLError:
                raise RuntimeError("Error downloading resource!")
            finally:
                print()
    except KeyboardInterrupt:
        print("Interrupted")

    # load
    print("Loading MNIST dataset.")
    with gzip.open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main(args):
    train_set, valid_set, test_set = prepare_dataset(args.data_dir)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = get_one_hot(train_y, 10)

    if args.model_type == "cnn":
        train_x = train_x.reshape((-1, 28, 28, 1))
        test_x = test_x.reshape((-1, 28, 28, 1))

    if args.model_type == "cnn":
        net = Net([
            Conv2D(kernel=[5, 5, 1, 8], stride=[2, 2], padding="SAME"),
            ReLU(),
            Conv2D(kernel=[5, 5, 8, 16], stride=[2, 2], padding="SAME"),
            ReLU(),
            Conv2D(kernel=[5, 5, 16, 32], stride=[2, 2], padding="SAME"),
            ReLU(),
            Flatten(),
            Dense(512, 10)
        ])
    elif args.model_type == "dense":
        net = Net([
            Dense(784, 200),
            ReLU(),
            Dense(200, 100),
            ReLU(),
            Dense(100, 70),
            ReLU(),
            Dense(70, 30),
            ReLU(),
            Dense(30, 10)
        ])
    else:
        raise ValueError("Invalid argument model_type! Must be 'cnn' or 'dense'")

    loss_fn = CrossEntropyLoss()

    if args.optim == "adam":
        optimizer = Adam(lr=args.lr)
    elif args.optim == "sgd":
        optimizer = SGD(lr=args.lr)
    elif args.optim == "momentum":
        optimizer = Momentum(lr=args.lr)
    elif args.optim == "rmsprop":
        optimizer = RMSProp(lr=args.lr)
    else:
        raise ValueError("Invalid Optimizer!!")

    model = Model(net=net, loss=loss_fn, optimizer=optimizer)
    model.initialize()
    # model.load("../data/model.pk")

    iterator = BatchIterator(batch_size=args.batch_size)
    evaluator = AccEvaluator()
    loss_list = list()
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_x, train_y):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grad(grads)
            loss_list.append(loss)
        print("Epoch %d time cost: %.4f" % (epoch, time.time() - t_start))
        # evaluate
        model.set_phase("TEST")
        test_pred = model.forward(test_x)
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.asarray(test_y)
        res = evaluator.evaluate(test_pred_idx, test_y_idx)
        print(res)
        model.set_phase("TRAIN")
    # model.save("../data/model.pk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="cnn", type=str, help="cnn or dense")
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--optim", default="adam", type=str)
    parser.add_argument("--lr", default=3e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)
