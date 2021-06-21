"""Implementation of asynchronous and synchronous SGD."""

import argparse
import copy
import os
import time

import numpy as np
import ray
import tinynn as tn


def get_model(lr):
    net = tn.net.Net([
        tn.layer.Dense(200),
        tn.layer.ReLU(),
        tn.layer.Dense(100),
        tn.layer.ReLU(),
        tn.layer.Dense(70),
        tn.layer.ReLU(),
        tn.layer.Dense(30),
        tn.layer.ReLU(),
        tn.layer.Dense(10)])
    model = tn.model.Model(net=net, loss=tn.loss.SoftmaxCrossEntropy(),
                           optimizer=tn.optimizer.Adam(lr=lr))
    model.net.init_params(input_shape=(784,))
    return model


@ray.remote
class ParamServer:

    def __init__(self, model, test_set):
        self.test_set = test_set
        self.model = model

    def get_params(self):
        return self.model.net.params

    def apply_grads(self, grads):
        self.model.apply_grads(grads)


@ray.remote
class Worker:

    def __init__(self, model, train_set):
        self.model = model
        self.train_set = train_set

        self.iter = tn.data_iterator.BatchIterator(batch_size=args.batch_size)
        self.batch_gen = None

    def get_next_batch(self):
        # reset batch generator if needed
        if self.batch_gen is None:
            self.batch_gen = self.iter(*self.train_set)

        try:
            batch = next(self.batch_gen)
        except StopIteration:
            self.batch_gen = None
            batch = self.get_next_batch()
        return batch

    def compute_grads(self):
        batch = self.get_next_batch()
        preds = self.model.forward(batch.inputs)
        _, grads = self.model.backward(preds, batch.targets)
        return grads

    def set_params(self, params):
        self.model.net.params = params


def main():
    if args.seed >= 0:
        tn.seeder.random_seed(args.seed)

    # data preparation
    mnist = tn.dataset.MNIST(args.data_dir, one_hot=True)

    # init ray
    ray.init()

    # init model
    model = get_model(args.lr)

    # init parameter server and workers
    ps = ParamServer.remote(model=copy.deepcopy(model),
                            test_set=mnist.test_set)
    workers = []
    for _ in range(1, args.num_workers + 1):
        worker = Worker.remote(model=copy.deepcopy(model),
                               train_set=mnist.train_set)
        workers.append(worker)

    start_time = time.time()
    iter_each_epoch = len(mnist.train_set[0]) // args.batch_size + 1
    iterations = args.num_ep * iter_each_epoch
    if args.mode == "async":
        print("Run asynchronous training.")
        # Workers repeatedly fetch global parameters, train one batch locally
        # and send their local gradients to the parameter server.
        # The parameter server updates global parameters once it receives
        # gradients from any workers.
        for i in range(iterations):
            global_params = ps.get_params.remote()
            for worker in workers:
                worker.set_params.remote(global_params)
                # compute local grads
                grads = worker.compute_grads.remote()
                # update global model asynchronously
                ps.apply_grads.remote(grads)

            # evaluate
            model.net.params = ray.get(ps.get_params.remote())
            acc = evaluate(mnist.test_set, model)
            print(f"[{time.time() - start_time}s] "
                  f"Accuracy after {i + 1} iterations: {acc:.4f}")
    elif args.mode == "sync":
        print("Run synchronous training.")
        # In each iteration, workers request for the global model,
        # compute local gradients and then send to the parameter server.
        # The parameter server gathers grads from all workers,
        # updates the global model and broadcasts to workers synchronously.
        for i in range(iterations):
            all_grads = []
            global_params = ps.get_params.remote()
            for worker in workers:
                # grab global params
                worker.set_params.remote(global_params)
                # compute local grads
                grads = worker.compute_grads.remote()
                all_grads.append(grads)

            # gathers grads from all workers
            all_grads = ray.get(all_grads)
            # update global model
            ps.apply_grads.remote(sum(all_grads))

            # evaluate
            model.net.params = ray.get(ps.get_params.remote())
            acc = evaluate(mnist.test_set, model)
            print(f"[{time.time() - start_time}s] "
                  f"Accuracy after {i + 1} iterations: {acc:.4f}")
    else:
        raise ValueError("Invalid train mode. Suppose to be 'sync' or 'async'.")


def evaluate(test_set, model):
    test_x, test_y = test_set
    test_pred = model.forward(test_x)

    test_pred_idx = np.argmax(test_pred, axis=1)
    test_y_idx = np.argmax(test_y, axis=1)
    acc, _ = tn.metric.accuracy(test_pred_idx, test_y_idx)
    return acc


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sync", type=str,
                        help="Train mode [sync|async]. Defaults to 'sync'.")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers.")
    parser.add_argument("--num_ep", default=3, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main()
