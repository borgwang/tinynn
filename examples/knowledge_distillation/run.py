"""Example code for knowledge distillation task."""

import argparse
import os
import time

import numpy as np
import tinynn as tn

from nets import student_net
from nets import teacher_net


class DistillationLoss(tn.loss.Loss):

    def __init__(self, alpha, T):
        self.alpha = alpha

        self.ce_loss = tn.loss.SoftmaxCrossEntropy()
        self.ce_loss_t = tn.loss.SoftmaxCrossEntropy(T=T)

    def loss(self, pred, label, teacher_prob):
        student_loss = self.ce_loss.loss(pred, label)
        distill_loss = self.ce_loss_t.loss(pred, teacher_prob)
        return self.alpha * distill_loss + (1 - self.alpha) * student_loss

    def grad(self, pred, label, teacher_prob):
        student_grad = self.ce_loss.grad(pred, label)
        distill_grad = self.ce_loss_t.grad(pred, teacher_prob)
        return self.alpha * distill_grad + (1 - self.alpha) * student_grad


def prepare_dataset(data_dir):
    fashion = tn.dataset.FashionMNIST(data_dir, one_hot=True)
    train_x, train_y = fashion.train_set
    test_x, test_y = fashion.test_set
    train_x = train_x.reshape((-1, 28, 28, 1))
    test_x = test_x.reshape((-1, 28, 28, 1))
    return train_x, train_y, test_x, test_y


def train_single_model(model, dataset, name="teacher"):
    print(f"training {name} model")
    train_x, train_y, test_x, test_y = dataset

    iterator = tn.data_iterator.BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        t_start = time.time()

        for i, batch in enumerate(iterator(train_x, train_y)):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grads(grads)
        print(f"Epoch {epoch} time cost: {time.time() - t_start}")

        # evaluate
        model.is_training = False
        for i, batch in enumerate(iterator(test_x, test_y)):
            pred = model.forward(batch.inputs)
            accuracy, _ = tn.metric.accuracy(
                np.argmax(pred, 1), np.argmax(batch.targets, 1))
        print(f"Accuracy: {accuracy:.4f}")
        model.is_training = True

    # save model
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = os.path.join(args.model_dir, name + ".model")
    model.save(model_path)
    print(f"Model saved in {model_path}")


def train_distill_model(dataset):
    # load dataset
    train_x, train_y, test_x, test_y = dataset
    # load or train a teacher model
    teacher = tn.model.Model(net=teacher_net,
                             loss=tn.loss.SoftmaxCrossEntropy(),
                             optimizer=tn.optimizer.Adam(lr=args.lr))
    teacher_model_path = os.path.join(args.model_dir, "teacher.model")
    if not os.path.isfile(teacher_model_path):
        print("No teacher model founded. Training a new one...")
        train_single_model(teacher, dataset, name="teacher")
    teacher.load(teacher_model_path)
    teacher.is_training = False

    print("training distill model")
    # define a student model
    student = tn.model.Model(net=student_net,
                             loss=DistillationLoss(alpha=args.alpha, T=args.T),
                             optimizer=tn.optimizer.Adam(lr=args.lr))

    # run training
    iterator = tn.data_iterator.BatchIterator(batch_size=args.batch_size)
    for epoch in range(args.num_ep):
        t_start = time.time()
        for batch in iterator(train_x, train_y):
            pred = student.forward(batch.inputs)
            teacher_out = teacher.forward(batch.inputs)
            teacher_out_prob = tn.math.softmax(teacher_out, t=args.T)

            grad_from_loss = student.loss.grad(
                pred, batch.targets, teacher_out_prob)
            grads = student.net.backward(grad_from_loss)
            student.apply_grads(grads)
        print(f"Epoch {epoch} time cost: {time.time() - t_start}")
        # evaluate
        student.is_trianing = False
        for batch in iterator(test_x, test_y):
            pred = student.forward(batch.inputs)
            accuracy, _ = tn.metric.accuracy(
                np.argmax(pred, 1), np.argmax(batch.targets, 1))
        print(f"Accuracy: {accuracy:.4f}")
        student.is_trianing = True

    # save the distilled model
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = os.path.join(args.model_dir, "distill-%d.model" % args.T)
    student.save(model_path)
    print(f"Model saved in {model_path}")


def main():
    if args.seed >= 0:
        tn.seeder.random_seed(args.seed)

    dataset = prepare_dataset(args.data_dir)

    if args.train_teacher:
        model = tn.model.Model(net=teacher_net,
                               loss=tn.loss.SoftmaxCrossEntropy(),
                               optimizer=tn.optimizer.Adam(lr=args.lr))
        train_single_model(model, dataset, name="teacher")

    if args.train_student:
        model = tn.model.Model(net=student_net,
                               loss=tn.loss.SoftmaxCrossEntropy(),
                               optimizer=tn.optimizer.Adam(lr=args.lr))
        train_single_model(model, dataset, name="student")

    train_distill_model(dataset)


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
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)

    parser.add_argument("--train_student", action="store_true")
    parser.add_argument("--train_teacher", action="store_true")
    parser.add_argument("--T", default=20.0, type=float)
    parser.add_argument("--alpha", default=0.9, type=float)
    args = parser.parse_args()
    main()
