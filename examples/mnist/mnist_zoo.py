# Author: borgwang <borgwang@126.com>
# Date: 2018-05-22
#
# Filename: mnist_zoo.py
# Description: mnist model zoo


from core.layers import Linear, Sigmoid, ReLU


ZOO = {
    "linear": [
        Linear(num_in=784, num_out=10)
    ],
    "5-layers-sigmoid": [
        Linear(num_in=784, num_out=200),
        Sigmoid(),
        Linear(num_in=200, num_out=100),
        Sigmoid(),
        Linear(num_in=100, num_out=60),
        Sigmoid(),
        Linear(num_in=60, num_out=30),
        Sigmoid(),
        Linear(num_in=30, num_out=10)
    ],
    "5-layers-relu": [
        Linear(num_in=784, num_out=200),
        ReLU(),
        Linear(num_in=200, num_out=100),
        ReLU(),
        Linear(num_in=100, num_out=60),
        ReLU(),
        Linear(num_in=60, num_out=30),
        ReLU(),
        Linear(num_in=30, num_out=10)
    ],
}
