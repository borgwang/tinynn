"""Example code for MNIST classification."""

import argparse
import os
import time

import numpy as np
from tinynn.core.layer import RNN
from tinynn.core.layer import Conv2D
from tinynn.core.layer import Dense
from tinynn.core.layer import Flatten
from tinynn.core.layer import MaxPool2D
from tinynn.core.layer import ReLU
from tinynn.core.layer import Tanh
from tinynn.core.loss import SoftmaxCrossEntropy
from tinynn.core.model import Model
from tinynn.core.net import Net
from tinynn.core.optimizer import Adam
from tinynn.utils.data_iterator import BatchIterator
from tinynn.utils.dataset import mnist
from tinynn.utils.metric import accuracy
from tinynn.utils.seeder import random_seed


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    train_set, _, test_set = mnist(args.data_dir, one_hot=True)
    train_x, train_y = train_set
    test_x, test_y = test_set

    if args.model_type == "mlp":
        # A multilayer perceptron model
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
    elif args.model_type == "cnn":
        # A LeNet-5 model with activation function changed to ReLU
        train_x = train_x.reshape((-1, 28, 28, 1))
        test_x = test_x.reshape((-1, 28, 28, 1))
        net = Net([
            Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1]),
            ReLU(),
            MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
            Conv2D(kernel=[5, 5, 6, 16], stride=[1, 1]),
            ReLU(),
            MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
            Flatten(),
            Dense(120),
            ReLU(),
            Dense(84),
            ReLU(),
            Dense(10)
        ])
    elif args.model_type == "rnn":
        # A simple recurrent neural net to classify images.
        train_x = train_x.reshape((-1, 28, 28))
        test_x = test_x.reshape((-1, 28, 28))
        net = Net([
            RNN(num_hidden=50, activation=Tanh()),
            Dense(10)
        ])
    else:
        raise ValueError("Invalid argument: model_type")

    model = Model(net=net, loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=args.lr))

    iterator = BatchIterator(batch_size=args.batch_size)
    loss_list = list()
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_x, train_y):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grads(grads)
            loss_list.append(loss)
        print("Epoch %d time cost: %.4f" % (epoch, time.time() - t_start))
        # evaluate
        model.set_phase("TEST")
        test_pred = model.forward(test_x)
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.argmax(test_y, axis=1)
        res = accuracy(test_pred_idx, test_y_idx)
        print(res)
        model.set_phase("TRAIN")
    
    # save model
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    model_name = "mnist-%s-epoch%d.pkl" % (args.model_type, args.num_ep)
    model_path = os.path.join(args.model_dir, model_name) 
    model.save(model_path)
    print("model saved in %s" % model_path)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--model_dir", type=str,
                        default=os.path.join(curr_dir, "models"))
    parser.add_argument("--model_type", default="mlp", type=str,
                        help="[*mlp|cnn|rnn]")
    parser.add_argument("--num_ep", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
