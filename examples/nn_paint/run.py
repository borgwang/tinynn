"""Use a neural network to mimic images."""

import argparse
import os

import numpy as np
from PIL import Image
from tinynn.core.layer import Dense
from tinynn.core.layer import ReLU
from tinynn.core.layer import Sigmoid
from tinynn.core.loss import MSE
from tinynn.core.model import Model
from tinynn.core.net import Net
from tinynn.core.optimizer import Adam
from tinynn.utils.data_iterator import BatchIterator
from tinynn.utils.metric import mean_square_error
from tinynn.utils.seeder import random_seed


def prepare_dataset(img_path):
    if not os.path.isfile(img_path):
        raise FileExistsError("Image %s not exist" % img_path)
    img = np.asarray(Image.open(img_path), dtype="float32") / 255.0

    train_x, train_y = [], []
    h, w, _ = img.shape
    for r in range(h):
        for c in range(w):
            train_x.append([(r - h / 2.0) / h, (c - w / 2.0) / w])
            train_y.append(img[r][c])
    return np.asarray(train_x), np.asarray(train_y), (h, w)


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    # data preparing
    train_x, train_y, img_shape = prepare_dataset(args.img)

    net = Net([
        Dense(30),
        ReLU(),
        Dense(100),
        ReLU(),
        Dense(100),
        ReLU(),
        Dense(30),
        ReLU(),
        Dense(3),
        Sigmoid()
    ])

    model = Model(net=net, loss=MSE(), optimizer=Adam())
    iterator = BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        for batch in iterator(train_x, train_y):
            preds = model.forward(batch.inputs)
            loss, grads = model.backward(preds, batch.targets)
            model.apply_grads(grads)

        # evaluate
        preds = net.forward(train_x)
        mse = mean_square_error(preds, train_y)
        print("Epoch %d %s" % (epoch, mse))

        # generate painting
        if epoch % 5 == 0:
            preds = preds.reshape(img_shape[0], img_shape[1], -1)
            preds = (preds * 255.0).astype("uint8")
            name, ext = os.path.splitext(args.img)
            filename = os.path.basename(name)
            out_filename = filename + "-paint-epoch" + str(epoch) + ext
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            out_path = os.path.join(args.output_dir, out_filename)
            Image.fromarray(preds).save(out_path)
            print("save painting to %s" % out_path)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str,
                        default=os.path.join(curr_dir, "test-img.jpg"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(curr_dir, "output"))
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_ep", default=100, type=int)
    main(parser.parse_args())
