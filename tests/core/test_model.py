import copy

import numpy as np
import pytest

from tinynn.core.layer import BatchNormalization
from tinynn.core.layer import Dense
from tinynn.core.layer import ReLU
from tinynn.core.loss import MSE
from tinynn.core.model import Model
from tinynn.core.net import Net
from tinynn.core.optimizer import Adam
from tinynn.utils.metric import mean_square_error
from tinynn.utils.seeder import random_seed

random_seed(0)


@pytest.fixture(name="model")
def model_fixture():
    net = Net([
        Dense(10),
        ReLU(),
        BatchNormalization(),
        Dense(1)
    ])
    return Model(net=net, loss=MSE(), optimizer=Adam(lr=0.01))


def test_model_save_and_load(model, tmpdir):
    model_copy = copy.deepcopy(model)
    # simple training
    x = np.random.uniform(0., 1, (100, 3))
    y = np.random.uniform(0., 1., (100, 1))

    for _ in range(5):
        pred = model.forward(x)
        _, grads = model.backward(pred, y)
        model.apply_grads(grads)

    pred = model.forward(x)
    mse_save = mean_square_error(pred, y)
    save_path = tmpdir.join("test-model.pkl")
    model.save(save_path)

    model_copy.load(save_path)
    pred = model_copy.forward(x)
    # same parameters
    for layer_save, layer_load in zip(model.net.layers, model_copy.net.layers):
        param_save, param_load = layer_save.params, layer_load.params
        assert param_save.keys() == param_load.keys()
        for key in param_save.keys():
            assert (param_save[key] == param_load[key]).all()
    # same mean_square_error
    mse_load = mean_square_error(pred, y)
    assert mse_save == mse_load
