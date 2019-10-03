"""tinynn example on a binary classification task."""

import runtime_path  # isort:skip

import argparse
import os
import sys
import time

import numpy as np

from core.evaluator import AccEvaluator
from core.layer import Dense
from core.layer import ReLU
from core.loss import SigmoidCrossEntropy
from core.model import Model
from core.net import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.seeder import random_seed


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def prepare_dataset(data_dir):
    # download dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt"
    save_path = os.path.join(data_dir, url.split("/")[-1])
    try:
        download_url(url, save_path)
    except Exception as e:
        print('Error downloading dataset: %s' % str(e))
        sys.exit(1)
    # read the dataset
    data = list()
    for line in open(save_path, "r").readlines():
        data.append(line.strip("\n").split("\t"))
    data = np.asarray(data).astype(float)

    n_samples = len(data)
    random_idx = np.arange(0, n_samples)
    np.random.shuffle(random_idx)
    data = data[random_idx]
    x, y = data[:, :3], data[:, 3:]
    y = (y - 1).astype(int)
    train_split = int(n_samples * 0.7)
    valid_split = int(n_samples * 0.85)
    train_set = [x[:train_split, :], y[:train_split]]
    valid_set = [x[train_split:valid_split, :], y[train_split:valid_split]]
    test_set = [x[valid_split:, :], y[valid_split:]]
    return [train_set, valid_set, test_set]


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    train_set, valid_set, test_set = prepare_dataset(args.data_dir)
    train_x, train_y = train_set
    test_x, test_y = test_set
    # train_y = get_one_hot(train_y, 2)

    net = Net([
        Dense(100),
        ReLU(),
        Dense(30),
        ReLU(),
        Dense(1)
    ])

    model = Model(net=net, loss=SigmoidCrossEntropy(), optimizer=Adam(lr=args.lr))

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
        test_y_idx = np.asarray(test_y).reshape(-1)
        test_pred = model.forward(test_x)
        test_pred[test_pred > 0] = 1
        test_pred[test_pred <= 0] = 0
        test_pred_idx = test_pred.reshape(-1)
        res = evaluator.evaluate(test_pred_idx, test_y_idx)
        print(res)
        model.set_phase("TRAIN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/binary_classification/data",
                        type=str, help="dataset directory")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
