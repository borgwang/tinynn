"""
Example code for a denoising autoencoder (DAE).
"""

import argparse
import os
import sys

import numpy as np
import tinynn as tn
from matplotlib import pyplot as plt

from autoencoder import AutoEncoder


def save_batch_as_images(path, batch, titles=None):
    m = batch.shape[0]  # batch size
    batch_copy = batch[:]
    batch_copy.resize(m, 28, 28)
    fig, ax = plt.subplots(int(m / 16), 16, figsize=(28, 28))
    cnt = 0
    for i in range(int(m / 16)):
        for j in range(16):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].imshow(batch_copy[cnt], cmap="gray",
                            interpolation="nearest", vmin=0, vmax=1)
            if titles is not None:
                ax[i][j].set_title(titles[cnt], fontsize=20)
            cnt += 1
    print(f"Saving {path}")
    plt.savefig(path)
    plt.close(fig)


def transition(code1, code2, n):
    """
    Make intermediate latent-space transition
    from code1 to code2 in n steps.
    """
    steps = (code2 - code1) / (n - 1)
    c = code1.copy()
    for _ in range(n):
        yield c
        c += steps  # towards code2 ...


def main():
    if args.seed >= 0:
        tn.utils.seeder.random_seed(args.seed)

    # create output directory for saving result images
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # prepare and read dataset
    mnist = tn.dataset.MNIST(args.data_dir)
    train_x, train_y = mnist.train_set
    test_x, test_y = mnist.test_set

    # specify the encoder and decoder net structure
    encoder_net = tn.net.Net([
        tn.layer.Dense(256),
        tn.layer.ReLU(),
        tn.layer.Dense(64)
    ])
    decoder_net = tn.net.Net([
        tn.layer.ReLU(),
        tn.layer.Dense(256),
        tn.layer.Tanh(),
        tn.layer.Dense(784),
        tn.layer.Tanh()
    ])
    nets = (encoder_net, decoder_net)
    optimizers = (tn.optimizer.Adam(args.lr), tn.optimizer.Adam(args.lr))
    loss = tn.loss.MSE()
    model = AutoEncoder(nets, loss=loss, optimizer=optimizers)

    # for pre-trained model, test generated images from latent space
    if args.load_model is not None:
        # load pre-trained model
        model.load(os.path.join(args.output_dir, args.load_model))
        print(f"Loaded model from {args.load_model}")

        # transition from test[from_idx] to test[to_idx] in n steps
        idx_arr, n = [2, 4, 32, 12, 82], 160
        print(f"Transition in numbers {[test_y[i] for i in idx_arr]}"
              f"in {n} steps ...")
        stops = [model.en_net.forward(test_x[i]) for i in idx_arr]
        k = int(n / (len(idx_arr) - 1))  # number of code per transition
        # generate all transition codes
        code_arr = []
        for i in range(len(stops) - 1):
            t = [c.copy() for c in transition(stops[i], stops[i+1], k)]
            code_arr += t
        # apply decoding all n "code" from latent space...
        batch = None
        for code in code_arr:
            # translate latent space to image
            genn = model.de_net.forward(code)
            # save decoded results in a batch
            if batch is None:
                batch = np.array(genn)
            else:
                batch = np.concatenate((batch, genn))
        output_path = os.path.join(args.output_dir, "genn-latent.png")
        save_batch_as_images(output_path, batch)
        sys.exit()

    # train the auto-encoder
    iterator = tn.data_iterator.BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        for batch in iterator(train_x, train_y):
            origin_in = batch.inputs

            # make noisy inputs
            m = origin_in.shape[0]  # batch size
            mu = args.gaussian_mean  # mean
            sigma = args.gaussian_std  # standard deviation
            noises = np.random.normal(mu, sigma, (m, 784))
            noises_in = origin_in + noises  # noisy inputs

            # forward
            genn = model.forward(noises_in)
            # back-propagate
            loss, grads = model.backward(genn, origin_in)

            # apply gradients
            model.apply_grads(grads)
        print(f"Epoch: {epoch} Loss: {loss:.3f}")

        # save all the generated images and original inputs for this batch
        noises_in_path = os.path.join(
            args.output_dir, "ep%d-input.png" % epoch)
        genn_path = os.path.join(
            args.output_dir, "ep%d-genn.png" % epoch)
        save_batch_as_images(noises_in_path, noises_in, titles=batch.targets)
        save_batch_as_images(genn_path, genn, titles=batch.targets)

    # save the model after training
    model.save(os.path.join(args.output_dir, args.save_model))


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(curr_dir, "output"))
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--gaussian_mean", default=0.3, type=float)
    parser.add_argument("--gaussian_std", default=0.2, type=float)
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--save_model", default="model", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main()
