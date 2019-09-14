"""
Example code for a denoising autoencoder (DAE).
"""

import runtime_path  # isort:skip

import argparse
import gzip
import os
import pickle
import sys

import numpy as np

from core.layers import Dense
from core.layers import ReLU
from core.layers import Tanh
from core.losses import MSELoss
from core.model import AutoEncoder
from core.nn import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.seeder import random_seed

### will modulize this part later
from matplotlib import cm as cm
from matplotlib import pyplot as plt
def disp_mnist_array(arr, label='unknown'):
    arr_copy = arr[:]
    arr_copy.resize(28,28)
    fig, ax = plt.subplots(1)
    ax.imshow(arr_copy, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.text(0.5, 1.5, 'label: %s' % label, bbox={'facecolor': 'white'})
    plt.show()
###

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

    # prepare and read dataset
    train_set, valid_set, test_set = prepare_dataset(args.data_dir)
    train_x, train_y = train_set

    # batch iterator
    iterator = BatchIterator(batch_size=args.batch_size)

    # specify the encoder and decoder net structure
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

    # create AutoEncoder model
    model = AutoEncoder(encoder=encoder, decoder=decoder,
                        loss=MSELoss(), optimizer=Adam(args.lr))

    # train the autoencoder
    for epoch in range(args.num_ep):
        print('epoch %d ...' % epoch)
        for batch in iterator(train_x, train_y):
            origin_in = batch.inputs
            # make noisy inputs
            m = origin_in.shape[0] # batch size
            mu = args.guassian_mean # mean
            sigma = args.guassian_std # standard deviation
            noises = np.random.normal(mu, sigma, (m, 784))
            noises_in = origin_in + noises # noisy inputs
            # forward pass
            genn = model.forward(noises_in)
            loss, grads = model.backward(genn, origin_in)
            model.apply_grad(grads)
        print(loss)
        for i in range(3):
            print(batch.targets[i])
            disp_mnist_array(noises_in[i])
            disp_mnist_array(genn[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--guassian_mean", default=0.3, type=float)
    parser.add_argument("--guassian_std", default=0.2, type=float)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
