from core.layer import BatchNormalization
from core.layer import Conv2D
from core.layer import Dense
from core.layer import Flatten
from core.layer import MaxPool2D
from core.layer import ReLU
from core.net import Net


def conv_bn_relu(kernel):
    return [Conv2D(kernel=kernel, stride=(1, 1), padding="SAME"),
            BatchNormalization(), ReLU()]


def max_pool():
    return MaxPool2D(pool_size=(2, 2), stride=(2, 2), padding="SAME")


vgg16 = Net([
    *conv_bn_relu((3, 3, 3, 64)),
    *conv_bn_relu((3, 3, 64, 64)),
    max_pool(),

    *conv_bn_relu((3, 3, 64, 128)),
    *conv_bn_relu((3, 3, 128, 128)),
    max_pool(),

    *conv_bn_relu((3, 3, 128, 256)),
    *conv_bn_relu((3, 3, 256, 256)),
    *conv_bn_relu((3, 3, 256, 256)),
    max_pool(),

    *conv_bn_relu((3, 3, 256, 512)),
    *conv_bn_relu((3, 3, 512, 512)),
    *conv_bn_relu((3, 3, 512, 512)),
    max_pool(),

    *conv_bn_relu((3, 3, 512, 512)),
    *conv_bn_relu((3, 3, 512, 512)),
    *conv_bn_relu((3, 3, 512, 512)),
    max_pool(),

    Flatten(),
    Dense(512),
    ReLU(),
    Dense(10)
])
