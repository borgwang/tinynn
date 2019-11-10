"""Example code for knowledge distillation task."""

import argparse
import os
import time

import numpy as np
from tinynn.core.loss import Loss
from tinynn.core.loss import SoftmaxCrossEntropy
from tinynn.core.model import Model
from tinynn.core.optimizer import Adam
from tinynn.utils.data_iterator import BatchIterator
from tinynn.utils.dataset import fashion_mnist
from tinynn.utils.math import softmax
from tinynn.utils.metric import accuracy
from tinynn.utils.seeder import random_seed

from nets import student_net
from nets import teacher_net


class DistillationLoss(Loss):

    def __init__(self, alpha, T):
        self.alpha = alpha

        self.ce_loss = SoftmaxCrossEntropy()
        self.ce_loss_t = SoftmaxCrossEntropy(T=T)

    def loss(self, pred, label, teacher_prob):
        student_loss = self.ce_loss.loss(pred, label)
        distill_loss = self.ce_loss_t.loss(pred, teacher_prob)
        return (self.alpha * distill_loss + 
                (1 - self.alpha) * student_loss)

    def grad(self, pred, label, teacher_prob):
        student_grad = self.ce_loss.grad(pred, label)
        distill_grad = self.ce_loss_t.grad(pred, teacher_prob)
        return (self.alpha * distill_grad + 
                (1 - self.alpha) * student_grad)


def prepare_dataset(data_dir):
    train_set, _, test_set = fashion_mnist(data_dir, one_hot=True)
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_x = train_x.reshape((-1, 28, 28, 1))
    test_x = test_x.reshape((-1, 28, 28, 1))
    return train_x, train_y, test_x, test_y


def train_single_model(model, dataset, args, name="teacher"):
    print("training %s model" % name)
    train_x, train_y, test_x, test_y = dataset

    train_iterator = BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        t_start = time.time()
        for i, batch in enumerate(train_iterator(train_x, train_y)):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grads(grads)

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
    model_path = os.path.join(args.model_dir, name + ".model") 
    model.save(model_path)
    print("model saved in %s" % model_path)


def train_distill_model(dataset, args):
    print("training distill model")
    # load dataset
    train_x, train_y, test_x, test_y = dataset

    # load or train a teacher model
    teacher = Model(net=teacher_net, 
                    loss=SoftmaxCrossEntropy(), 
                    optimizer=Adam(lr=args.lr))
    teacher_model_path = os.path.join(args.model_dir, "teacher.model") 
    if not os.path.isfile(teacher_model_path):
        print("No teacher model founded. Training a new one...")
        train_single_model(teacher, dataset, args, name="teacher")
    teacher.load(teacher_model_path)

    # define a student model
    student = Model(net=student_net, 
                    loss=DistillationLoss(alpha=args.alpha, T=args.T),
                    optimizer=Adam(lr=args.lr))

    # run training
    iterator = BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        t_start = time.time()
        for i, batch in enumerate(iterator(train_x, train_y)):
            pred = student.forward(batch.inputs)
            teacher_out = teacher.forward(batch.inputs)
            teacher_out_prob = softmax(teacher_out, t=args.T)

            loss = student.loss.loss(pred, batch.targets, teacher_out_prob)
            grad_from_loss = student.loss.grad(pred, batch.targets, teacher_out)
            grads = student.net.backward(grad_from_loss)
            student.apply_grads(grads)

        print("Epoch %d time cost: %.4f" % (epoch, time.time() - t_start))
        # evaluate
        student.set_phase("TEST")
        test_pred = student.forward(test_x)
        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.argmax(test_y, axis=1)
        res = accuracy(test_pred_idx, test_y_idx)
        print(res)
        student.set_phase("TRAIN")
    
    # save the distilled model
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = os.path.join(args.model_dir, "distill-%d.model" % args.T) 
    student.save(model_path)
    print("model saved in %s" % model_path)


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    dataset = prepare_dataset(args.data_dir)

    if args.train_teacher:
        model = Model(net=teacher_net, 
                      loss=SoftmaxCrossEntropy(),
                      optimizer=Adam(lr=args.lr))
        train_single_model(model, dataset, args, name="teacher")

    if args.train_student:
        model = Model(net=student_net,
                      loss=SoftmaxCrossEntropy(),
                      optimizer=Adam(lr=args.lr))
        train_single_model(model, dataset, args, name="student")

    train_distill_model(dataset, args)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--model_dir", type=str,
                        default=os.path.join(curr_dir, "models"))
    parser.add_argument("--model_type", default="cnn", type=str,
                        help="[*cnn]")

    parser.add_argument("--num_ep", default=10, type=int)
    parser.add_argument("--lr", default=7.5e-4, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)

    parser.add_argument("--train_student", action="store_true")
    parser.add_argument("--train_teacher", action="store_true")
    parser.add_argument("--T", default=20.0, type=float)
    parser.add_argument("--alpha", default=0.9, type=float)
    args = parser.parse_args()
    main(args)
