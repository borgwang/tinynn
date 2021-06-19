import tinynn as tn


def G_mlp():
    w_init = tn.initializer.Normal(0.0, 0.02)
    return tn.net.Net([
        tn.layer.Dense(100, w_init=w_init),
        tn.layer.LeakyReLU(),
        tn.layer.Dense(300, w_init=w_init),
        tn.layer.LeakyReLU(),
        tn.layer.Dense(784, w_init=w_init),
        tn.layer.Sigmoid()])


def D_mlp():
    w_init = tn.initializer.Normal(0.0, 0.02)
    return tn.net.Net([
        tn.layer.Dense(300, w_init=w_init),
        tn.layer.LeakyReLU(),
        tn.layer.Dense(100, w_init=w_init),
        tn.layer.LeakyReLU(),
        tn.layer.Dense(1, w_init=w_init)])


def G_cnn():
    return tn.net.Net([
        tn.layer.Dense(7 * 7 * 16),
        tn.layer.Reshape(7, 7, 16),
        tn.layer.ConvTranspose2D(kernel=[5, 5, 16, 6], stride=[2, 2],
                                 padding="SAME"),
        tn.layer.LeakyReLU(),
        tn.layer.ConvTranspose2D(kernel=[5, 5, 6, 1], stride=[2, 2],
                                 padding="SAME"),
        tn.layer.Sigmoid()])


def D_cnn():
    return tn.net.Net([
        tn.layer.Conv2D(kernel=[5, 5, 1, 6], stride=[1, 1], padding="SAME"),
        tn.layer.LeakyReLU(),
        tn.layer.MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        tn.layer.Conv2D(kernel=[5, 5, 6, 16], stride=[1, 1], padding="SAME"),
        tn.layer.LeakyReLU(),
        tn.layer.MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
        tn.layer.Flatten(),
        tn.layer.Dense(120),
        tn.layer.LeakyReLU(),
        tn.layer.Dense(84),
        tn.layer.LeakyReLU(),
        tn.layer.Dense(1)])
