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

from matplotlib import cm as cm
from matplotlib import pyplot as plt

def save_batch_as_images(path, batch, titles=None):
    m = batch.shape[0] # batch size
    batch_copy = batch[:]
    batch_copy.resize(m, 28, 28)
    fig, ax = plt.subplots(int(m / 16), 16, figsize=(28,28))
    for i in range(int(m/16)):
        for j in range(16):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].imshow(batch_copy[i*j], cmap='gray',
                interpolation='nearest', vmin=0, vmax=1)
            if titles is not None:
                ax[i][j].set_title(titles[i*j], fontsize=20)
    print('Saving', path)
    plt.savefig(path)
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

    # create output directory for saving result images
    if not os.path.exists('./output'):
        os.mkdir('./output')

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

    # for pretrained model, print the generated images from latent space
    if args.load_model is not None:
        model.load(args.load_model)
        print('Loaded model from %s' % args.load_model)
        for batch in iterator(train_x, train_y):
            origin_in = batch.inputs
            genn = model.forward(origin_in)
            save_batch_as_images('output/genn.png',
                genn, titles=batch.targets)
            break
        quit()

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
            # train the representation
            genn = model.forward(noises_in)
            loss, grads = model.backward(genn, origin_in)
            model.apply_grad(grads)
        print('Loss: %.3f' % loss)
        # save all the generated images and original inputs for this batch
        save_batch_as_images('output/ep%d-input.png' % epoch,
            noises_in, titles=batch.targets)
        save_batch_as_images('output/ep%d-genn.png' % epoch,
            genn, titles=batch.targets)

    # save the model after training
    model.save('output/model.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--guassian_mean", default=0.3, type=float)
    parser.add_argument("--guassian_std", default=0.2, type=float)
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
