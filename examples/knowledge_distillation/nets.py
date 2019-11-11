from tinynn.core.layer import Conv2D
from tinynn.core.layer import Dense
from tinynn.core.layer import Flatten
from tinynn.core.layer import MaxPool2D
from tinynn.core.layer import ReLU
from tinynn.core.net import Net
from tinynn.core.layer import BatchNormalization


def conv_bn_relu(kernel):
    return [Conv2D(kernel=kernel, stride=(1, 1), padding="SAME"),
            BatchNormalization(), ReLU()]


def max_pool():
    return MaxPool2D(pool_size=(2, 2), stride=(2, 2), padding="SAME")


teacher_net = Net([
    *conv_bn_relu((3, 3, 1, 64)),
    *conv_bn_relu((3, 3, 64, 64)),
    max_pool(),

    *conv_bn_relu((3, 3, 64, 128)),
    *conv_bn_relu((3, 3, 128, 128)),
    max_pool(),

    *conv_bn_relu((3, 3, 128, 256)),
    *conv_bn_relu((3, 3, 256, 256)),
    max_pool(),

    *conv_bn_relu((3, 3, 256, 512)),
    *conv_bn_relu((3, 3, 512, 512)),
    max_pool(),

    *conv_bn_relu((3, 3, 512, 512)),
    *conv_bn_relu((3, 3, 512, 512)),
    max_pool(),

    Flatten(),
    Dense(512),
    ReLU(),
    Dense(10)
])


student_net = Net([
    Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1]),
    ReLU(),
    Conv2D(kernel=[5, 5, 6, 12], stride=[1, 1]),
    ReLU(),
    Flatten(),
    Dense(10)])
