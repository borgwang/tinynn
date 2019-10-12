"""tinynn implementation of Deep Convolution Generative Adversarial Network."""

import runtime_path  # isort:skip

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from core.initializer import Normal
from core.layer import Dense
from core.layer import LeakyReLU
from core.layer import Sigmoid
from core.loss import SigmoidCrossEntropy
from core.model import Model
from core.net import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator
from utils.dataset import mnist
from utils.seeder import random_seed


def get_noise(size):
    return np.random.normal(size=size)


def mlp_G():
    w_init = Normal(0.0, 0.02)
    return Net([
        Dense(100, w_init=w_init), 
        LeakyReLU(), 
        Dense(300, w_init=w_init), 
        LeakyReLU(), 
        Dense(784, w_init=w_init), 
        Sigmoid()])


def mlp_D():
    w_init = Normal(0.0, 0.02)
    return Net([
        Dense(300, w_init=w_init), 
        LeakyReLU(), 
        Dense(100, w_init=w_init), 
        LeakyReLU(), 
        Dense(1, w_init=w_init)])


def train(args):
    # prepare dataset
    train_, valid, test = mnist(args.data_dir)
    X = np.concatenate([train_[0], valid[0], test[0]])
    y = np.concatenate([train_[1], valid[1], test[1]])

    fix_noise = get_noise(size=(args.batch_size, args.nz))

    loss = SigmoidCrossEntropy()
    # TODO: replace mlp with cnn
    G = Model(net=mlp_G(), loss=loss,
              optimizer=Adam(args.lr_g, beta1=args.beta1))
    D = Model(net=mlp_D(), loss=loss,
              optimizer=Adam(args.lr_d, beta1=args.beta1))

    running_g_err, running_d_err = 0, 0
    iterator = BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        for i, batch in enumerate(iterator(X, y)):
            # --- Train Discriminator ---
            # feed with real data (maximize log(D(x)))
            d_pred_real = D.forward(batch.inputs)
            label_real = np.ones_like(d_pred_real)
            d_real_err, d_real_grad = D.backward(
                d_pred_real, label_real)

            # feed with fake data (maximize log(1 - D(G(z))))
            noise = get_noise(size=(len(batch.inputs), args.nz))
            g_out = G.forward(noise)
            d_pred_fake = D.forward(g_out)
            label_fake = np.zeros_like(d_pred_fake)
            d_fake_err, d_fake_grad = D.backward(
                d_pred_fake, label_fake)

            # train D
            d_err = d_real_err + d_fake_err
            d_grads = d_real_grad + d_fake_grad
            D.apply_grad(d_grads)

            # ---- Train Generator ---
            # maximize log(D(G(z)))
            d_pred_fake = D.forward(g_out)
            g_err, d_grad = D.backward(d_pred_fake, label_real)
            g_grads = G.net.backward(d_grad.wrt_input)
            G.apply_grad(g_grads)

            running_d_err = 0.9 * running_d_err + 0.1 * d_err
            running_g_err = 0.9 * running_g_err + 0.1 * g_err
            if i % 100 == 0:
                print("epoch-%d iter-%d d_err: %.4f g_err: %.4f" %
                      (epoch+1, i+1, running_d_err, running_g_err))

        # sampling
        print("epoch: %d/%d d_err: %.4f g_err: %.4f" %
              (epoch+1, args.num_ep, running_d_err, running_g_err))
        samples = G.forward(fix_noise)
        img_name = "ep%d.png" % (epoch + 1)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_path = os.path.join(args.output_dir, img_name)
        save_batch_as_images(save_path, samples)

        # save generator
        model_path = os.path.join(args.output_dir, args.model_name)
        G.save(model_path)
        print("Saving generator ", model_path)


def evaluate(args):
    G = Model(net=mlp_G(), loss=None, optimizer=None)
    model_path = os.path.join(args.output_dir, args.model_name)
    print("Loading model from ", model_path)
    G.load(model_path)
    noise = get_noise(size=(128, args.nz))
    samples = G.forward(noise)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_path = os.path.join(args.output_dir, "evaluate.png")
    save_batch_as_images(save_path, samples)


def save_batch_as_images(path, batch, titles=None):
    m = batch.shape[0]  # batch size
    batch_copy = batch[:]
    batch_copy.resize(m, 28, 28)
    fig, ax = plt.subplots(int(m / 16), 16, figsize=(28, 28))
    cnt = 0
    for i in range(int(m/16)):
        for j in range(16):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].imshow(batch_copy[cnt], cmap="gray",
                            interpolation="nearest", vmin=0, vmax=1)
            if titles is not None:
                ax[i][j].set_title(titles[cnt], fontsize=20)
            cnt += 1
    print("Saving", path)
    plt.savefig(path)
    plt.close(fig)


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    if args.train:
        train(args)

    if args.evaluate:
        evaluate(args)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(curr_dir, "samples"))
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--model_name", type=str, default="generator.pkl")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--lr_g", default=7.5e-4, type=float)
    parser.add_argument("--lr_d", default=2e-4, type=float)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--nz", default=50, type=int,
                        help="dimension of latent z vector")

    main(parser.parse_args())
