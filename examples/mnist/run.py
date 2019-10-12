"""Example code for MNIST classification."""

import runtime_path  # isort:skip

import argparse
import os
import time

import numpy as np

from core.layer import Conv2D
from core.layer import Dense
from core.layer import Flatten
from core.layer import MaxPool2D
from core.layer import ReLU
from core.loss import SoftmaxCrossEntropy
from core.model import Model
from core.net import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator
from utils.dataset import mnist
from utils.metric import accuracy
from utils.seeder import random_seed


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    train_set, valid_set, test_set = mnist(args.data_dir)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_y = get_one_hot(train_y, 10)

    if args.model_type == "cnn":
        train_x = train_x.reshape((-1, 28, 28, 1))
        test_x = test_x.reshape((-1, 28, 28, 1))

    if args.model_type == "cnn":
        # a LeNet-5 model with activation function changed to ReLU
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

    model = Model(net=net, loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=args.lr))

    iterator = BatchIterator(batch_size=args.batch_size)
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
        res = accuracy(test_pred_idx, test_y_idx)
        print(res)
        model.set_phase("TRAIN")


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--model_type", default="dense", type=str,
                        help="cnn or dense")
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
