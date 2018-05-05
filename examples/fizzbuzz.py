import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from typing import List

from core.train import train
from core.nn import NeuralNet
from core.layers import Linear, Tanh
from core.optim import SGD


def fizzbuzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]


inputs = np.array([binary_encode(x) for x in range(101, 1024)])
targets = np.array([fizzbuzz_encode(x) for x in range(101, 1024)])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net, inputs, targets, num_epochs=5000, optimizer=SGD(lr=0.001))

for x in range(1, 101):
    predicted = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizzbuzz_encode(x))
    labels = [str(x), 'fizz', 'buzz', 'fizzbuzz']
    if predicted_idx != actual_idx:
        print('!!!!!!!!!!!!!')
    print('%10s %10s %10s' % (x, labels[predicted_idx], labels[actual_idx]))
