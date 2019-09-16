"""
Example code for visualizing feature maps and dense layers of
convolutional network using Activation Maximization (AM).
The subject Network is a pre-trained LeNet-5.

Reference
[1] Visualizing Higher-Layer Features of a Deep Network (Dumitru
    Erhan, et al.)
[2] Deep Inside Convolutional Networks: Visualising Image Classi-
    fication Models and Saliency Maps (Karen Simonyan, et al.)
"""

import runtime_path  # isort:skip

import argparse
import gzip
import os
import pickle

import numpy as np
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from core.layers import Conv2D
from core.layers import Dense
from core.layers import Flatten
from core.layers import MaxPool2D
from core.layers import ReLU
from core.losses import SoftmaxCrossEntropyLoss
from core.model import Model
from core.nn import Net
from core.optimizer import Adam
from utils.seeder import random_seed


def disp_mnist_batch(batch, fig=None):
    batch = batch[:]
    batch.resize(28, 28)
    if fig is None:
        fig = plt.figure()
        fig.show()
    ax = fig.gca()
    ax.imshow(batch, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    fig.canvas.draw()
    return fig


def activation_maximazation(model, init_grads, layer_idx, fig):
    # create a random image according to MNIST statistics
    img_mean = 0.456
    img_std = 0.224
    img = np.random.normal(img_mean, img_std, (1, 28, 28, 1))
    disp_mnist_batch(img, fig)
    # create optimizer
    opt = Adam(lr=1e-2)
    sigma = 0.30

    for iteration in range(500 + 1):
        # forward pass until interested layer
        outputs = img
        for layer in model.net.layers[0:layer_idx+1]:
            outputs = layer.forward(outputs)
        # backward from interested layer
        grads = init_grads
        for layer in model.net.layers[0:layer_idx+1][::-1]:
            grads = layer.backward(grads)
        # flatten the gradients and apply steps
        flat_grads = np.ravel(grads)
        flat_steps = opt._compute_step(flat_grads)
        steps = flat_steps.reshape(img.shape)
        img += steps
        # blur image to regularize this progress
        img = gaussian_filter(img, sigma, order=0)
        # ensure image is still in [0, 1] range
        mean, max_, min_ = img.mean(), img.max(), img.min()
        img = (img - min_) / (max_ - min_)

        if iteration % 100 == 0:
            cells_idx = (- init_grads).astype(int).astype(bool)
            loss = - outputs[cells_idx].sum()
            stats = img.mean(), img.std(), img.min(), img.max()
            print('Iteration#%d, loss: %.3f' % (iteration, loss), end=" ")
            print('image: u=%.3f, std=%.3f, range=(%.3f, %.3f)' % stats)
            disp_mnist_batch(img, fig)


def am_visualize_conv_layer(model, layer_idx, fig):
    # get size of layer-wise input gradients (or forward output size)
    grads = np.zeros(model.net.layers[layer_idx].cache['out_img_size'])
    grads = np.array([grads]) # adjust dimension
    n = grads.shape[3] # number of channels
    # for each feature map in this layer
    for idx in range(n):
        # fix the gradients for the cells we are interested to maximize
        fixed_grads = grads.copy()
        fixed_grads[:,:,:, idx] = -1
        # generate the image that maximizes the target cell(s)
        print('AM for feature map', idx)
        img = activation_maximazation(model, fixed_grads, layer_idx, fig)


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    # create output directory for saving result images
    if not os.path.exists('./output'): os.mkdir('./output')

    # define network we are going to load
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

    # load the model
    model = Model(
        net=net,
        loss=SoftmaxCrossEntropyLoss(),
        optimizer=Adam()
    )
    print('loading pre-trained model file', args.model_path)
    model.load(args.model_path)

    # create pyplot window for on-the-fly visualization
    img = np.ones((1, 28, 28, 1))
    fig = disp_mnist_batch(img)

    print('[ conv-layer-1 ]')
    am_visualize_conv_layer(model, 0, fig)

    print('[ conv-layer-2 ]')
    am_visualize_conv_layer(model, 3, fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./lenet-relu.pkl",
        help="pre-trained model file", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
