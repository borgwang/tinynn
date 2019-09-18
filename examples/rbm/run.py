"""
Example code for a Restricted Boltzmann Machine (RBM) training
using Contrastive Divergence (CD-k) algorithm.
"""

import runtime_path  # isort:skip

import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np

from core.layers import RBM
from utils.data_iterator import BatchIterator
from utils.downloader import download_url
from utils.seeder import random_seed

from matplotlib import cm as cm
from matplotlib import pyplot as plt
import math

def save_batch_as_images(path, batch, title=None, subs=None):
    m = batch.shape[0] # batch size
    batch_copy = batch[:]
    batch_copy.resize(m, 28, 28)
    w = math.floor(math.sqrt(m))
    h = math.ceil(m / float(w))
    fig, ax = plt.subplots(h, w, figsize=(28, 28))
    if title is not None: fig.suptitle(title, fontsize=60)
    cnt = 0
    while cnt < w * h:
        i, j = int(cnt / w), cnt % w
        if cnt < m:
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].imshow(batch_copy[cnt], cmap='gray',
                interpolation='nearest', vmin=0, vmax=1)
            if subs is not None:
                ax[i][j].set_title(subs[cnt], fontsize=40)
        else:
            ax[i, j].axis('off')
        cnt += 1
    print('Saving', path)
    plt.savefig(path, facecolor='grey')
    plt.close(fig)


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

    rbm = RBM(500, k=args.k)

    iterator = BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        print('epoch', epoch)
        for batch in iterator(train_x, train_y):
            visible = rbm.gibs_sampling(batch.inputs)
            rbm.step(args.lr)
        filename = './out/epoch-%d.png' % epoch
        save_batch_as_images(filename, visible, title='epoch %d' % epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="cnn", type=str,
        help="cnn or dense")
    parser.add_argument("--num_ep", default=30, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--lr", default=5e-2, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--k", default=1, type=int)
    args = parser.parse_args()
    main(args)
