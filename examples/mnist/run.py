"""Example code for MNIST. A fully-connected network and a convolutional neural network were implemented."""

import runtime_path  # isort:skip

import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np

from core.evaluator import AccEvaluator
from core.layers import Conv2D
from core.layers import Dense
from core.layers import Flatten
from core.layers import MaxPool2D
from core.layers import ReLU
from core.losses import SoftmaxCrossEntropyLoss
from core.model import Model
from core.nn import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.seeder import random_seed


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def prepare_dataset(data_dir):
    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    save_path = os.path.join(data_dir, url.split("/")[-1])
    print("Preparing MNIST dataset ...")
    try:
        download_url(url, save_path)
    except Exception as e:
        print('Error downloading dataset: %s' % str(e))
        sys.exit(1)
    # load the dataset
    with gzip.open(save_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    train_set, valid_set, test_set = prepare_dataset(args.data_dir)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = get_one_hot(train_y, 10)

    if args.model_type.startswith("cnn"):
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
            Dense(10)
        ])
    elif args.model_type == "cnn-lenet5":
        net = Net([
            Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1], padding="SAME"),
            ReLU(),
            MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
            Conv2D(kernel=[5, 5, 6, 16], stride=[1, 1], padding="SAME"),
            ReLU(),
            MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
            Flatten(),
            Dense(120),
            ReLU(),
            Dense(84),
            ReLU(),
            Dense(10)
        ])
    elif args.model_type == "dense":
        net = Net([
            Dense(200),
            ReLU(),
            Dense(100),
            ReLU(),
            Dense(70),
            ReLU(),
            Dense(30),
            ReLU(),
            Dense(10)
        ])
    else:
        raise ValueError("Invalid argument: model_type")

    model = Model(net=net, loss=SoftmaxCrossEntropyLoss(), optimizer=Adam(lr=args.lr))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="cnn", type=str,
        help="cnn, cnn-lenet5 or dense")
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
