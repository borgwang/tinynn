"""
Example code for a denoising autoencoder.
"""

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
from core.layers import ReLU
from core.layers import Tanh
from core.losses import MSELoss
from core.losses import SoftmaxCrossEntropyLoss
from core.model import Model
from core.nn import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.seeder import random_seed

from matplotlib import pyplot as plt
from matplotlib import cm as cm

def disp_mnist_array(arr, label='unknown'):
    arr_copy = arr[:]
    arr_copy.resize(28,28)
    fig, ax = plt.subplots(1)
    ax.imshow(arr_copy, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.text(0.5, 1.5, 'label: %s' % label, bbox={'facecolor': 'white'})
    plt.show()

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

    encoder = Net([
        Dense(256),
        ReLU(),
        Dense(64),
        ReLU()
    ])

    decoder = Net([
        Dense(256),
        Tanh(),
        Dense(784),
        Tanh()
    ])

    loss_fun = MSELoss()
    iterator = BatchIterator(batch_size=args.batch_size)

    decoder_opt = Adam(lr=args.lr)
    encoder_opt = Adam(lr=args.lr)

    for epoch in range(args.num_ep):
        print('epoch', epoch)
        for batch in iterator(train_x, train_y):
            origin_inputs = batch.inputs
            # make some noises
            m = origin_inputs.shape[0]
            noises = np.random.normal(0.3, 0.2, (m, 784))
            noises_inputs = origin_inputs + noises
            # forward pass
            code = encoder.forward(noises_inputs)
            genn = decoder.forward(code) # of shape (128, 784)
            # calculate loss
            loss = loss_fun.loss(genn, origin_inputs)
            grad = loss_fun.grad(genn, origin_inputs)
            # backward pass
            grads, grad = decoder.backward(grad)
            params = decoder.get_parameters()
            steps = decoder_opt.compute_step(grads, params)
            for step, param in zip(steps, params):
                for k, v in param.items():
                    param[k] += step[k]
            grads, grad = encoder.backward(grad)
            params = encoder.get_parameters()
            steps = encoder_opt.compute_step(grads, params)
            for step, param in zip(steps, params):
                for k, v in param.items():
                    param[k] += step[k]
        print('loss', loss)
        if epoch > 8:
            for i in range(3):
                print(batch.targets[i])
                disp_mnist_array(noises_inputs[i])
                disp_mnist_array(genn[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
