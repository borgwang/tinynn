import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from core.train import train, evaluate
from core.nn import NeuralNet
from core.layers import Linear, Tanh
from core.optimizer import SGD, Adam, RMSProp, Momentum
from core.initializer import NormalInit, UniformInit, ZerosInit, ConstantInit, XavierUniformInit, XavierNormalInit, OrthogonalInit, SparseInit

from core.data.data import DataIterator, BatchIterator
from core.data.dataset import MNIST


train = MNIST('../data', train=True, transform=None, batch_size=32, shuffle=True)
dataset = MNIST('../data', train=False, transform=None, batch_size=32, shuffle=True)


# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.test_batch_size, shuffle=True, **kwargs)
