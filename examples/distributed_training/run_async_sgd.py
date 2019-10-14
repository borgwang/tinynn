"""Example code for training neural networks with Asynchronous SGD."""

import runtime_path  # isort:skip

import argparse
import copy
import os
import time

import numpy as np
import ray

from core.layer import Dense
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


@ray.remote
class ParamServer(object):

    def __init__(self, model):
        self.model = model
        self.model.net.init_params(input_shape=(784,))

    def get_params(self):
        return self.model.net.params

    def apply_grads(self, grads):
        self.model.apply_grad(grads)


@ray.remote
class Worker(object):

    def __init__(self, rank, model, param_server, batch_size, num_ep):
        self.rank = rank
        self.param_server = param_server

        self.model = model
        self.batch_size = batch_size
        self.num_ep = num_ep

    def run(self, train_set):
        train_x, train_y = train_set
        iterator = BatchIterator(batch_size=self.batch_size)

        for epoch in range(1, self.num_ep + 1):
            for batch in iterator(train_x, train_y):
                # fetch model params from server
                params_id = self.param_server.get_params.remote()
                self.model.net.params = ray.get(params_id)

                # get local gradients
                preds = self.model.forward(batch.inputs)
                loss, grads = self.model.backward(preds, batch.targets)

                # send gradients back to server
                self.param_server.apply_grads.remote(grads)


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    # data preparation
    train_set, valid_set, test_set = mnist(args.data_dir)
    train_set = (train_set[0], get_one_hot(train_set[1], 10))

    # init model
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
    model = Model(net=net, loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=args.lr))

    # init ray
    ray.init(include_webui=False, ignore_reinit_error=True)

    # init parameter server and workers
    ps = ParamServer.remote(copy.deepcopy(model))
    workers = []
    for rank in range(1, args.num_workers + 1):
        worker = Worker.remote(rank=rank, 
                               model=copy.deepcopy(model), 
                               param_server=ps,
                               batch_size=args.batch_size,
                               num_ep=args.num_ep)
        workers.append(worker)

    # async run
    for worker in workers:
        worker.run.remote(train_set)

    for _ in range(100):
        model.net.params = ray.get(ps.get_params.remote())
        model.set_phase("TEST")

        test_x, test_y = test_set
        test_pred = model.forward(test_x)
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.asarray(test_y)
        res = accuracy(test_pred_idx, test_y_idx)
        print(res)

        time.sleep(3)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--num_workers", "-n", type=int, 
                        default=1, help="Number of workers.")
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    main(parser.parse_args())
