import tinynn as tn


__all__ = ["teacher_net", "student_net"]


def conv_bn_relu(kernel):
    return [tn.layer.Conv2D(kernel=kernel, stride=(1, 1), padding="SAME"), tn.layer.ReLU()]


def max_pool():
    return tn.layer.MaxPool2D(pool_size=(2, 2), stride=(2, 2), padding="SAME")


teacher_net = tn.net.Net([
    *conv_bn_relu((3, 3, 1, 32)),
    *conv_bn_relu((3, 3, 32, 32)),
    max_pool(),

    *conv_bn_relu((3, 3, 32, 64)),
    *conv_bn_relu((3, 3, 64, 64)),
    max_pool(),

    *conv_bn_relu((3, 3, 64, 128)),
    *conv_bn_relu((3, 3, 128, 128)),
    max_pool(),

    tn.layer.Flatten(),
    tn.layer.Dense(512),
    tn.layer.ReLU(),
    tn.layer.Dense(10)
])


student_net = tn.net.Net([
    tn.layer.Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1]),
    tn.layer.ReLU(),
    tn.layer.Conv2D(kernel=[5, 5, 6, 12], stride=[1, 1]),
    tn.layer.ReLU(),
    tn.layer.Flatten(),
    tn.layer.Dense(10)])
