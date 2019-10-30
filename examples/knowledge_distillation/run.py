"""Example code for MNIST classification."""

import runtime_path  # isort:skip

import argparse
import os
import time

import numpy as np

from core.loss import SoftmaxCrossEntropy
from core.model import Model
from core.optimizer import Adam
from examples.knowledge_distillation.nets import vgg16
from utils.data_iterator import BatchIterator
from utils.dataset import cifar10
from utils.metric import accuracy
from utils.seeder import random_seed


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    train_set, test_set = cifar10(args.data_dir, one_hot=True)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_x = train_x.reshape((-1, 32, 32, 3))
    test_x = train_x.reshape((-1, 32, 32, 3))

    model = Model(net=vgg16, loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=args.lr))

    train_iterator = BatchIterator(batch_size=args.batch_size)
    test_generator = BatchIterator(batch_size=args.batch_size)(test_x, test_y)
    running_loss = None
    for epoch in range(args.num_ep):
        t_start = time.time()
        for i, batch in enumerate(train_iterator(train_x, train_y)):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grads(grads)
            if running_loss is None:
                running_loss = loss
            else:
                running_loss = 0.9 * running_loss + 0.1 * loss
            print("Batch %d: Running loss: %.4f" % (i, running_loss))

        print("Epoch %d time cost: %.4f" % (epoch, time.time() - t_start))
        # evaluate
        model.set_phase("TEST")
        batch = next(test_generator)
        test_pred = model.forward(batch.inputs)
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.argmax(batch.targets, axis=1)
        res = accuracy(test_pred_idx, test_y_idx)
        print(res)
        model.set_phase("TRAIN")
    
    # save model
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    model_name = "cifar10-%s-epoch%d.pkl" % (args.model_type, args.num_ep)
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
    parser.add_argument("--model_type", default="vgg16", type=str,
                        help="[*vgg16]")
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
