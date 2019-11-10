from tinynn.core.layer import Conv2D
from tinynn.core.layer import Dense
from tinynn.core.layer import Flatten
from tinynn.core.layer import MaxPool2D
from tinynn.core.layer import ReLU
from tinynn.core.net import Net


teacher_net = Net([
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
    Dense(10)])


student_net = Net([
    Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1]),
    Flatten(),
    Dense(120),
    ReLU(),
    Dense(10)])
