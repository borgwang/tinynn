import copy

import numpy as np
import pytest
import tinynn as tn

tn.seeder.random_seed(31)


@pytest.fixture(name="model")
def model_fixture():
    net = tn.net.Net([
        tn.layer.Dense(10),
        tn.layer.ReLU(),
        tn.layer.BatchNormalization(),
        tn.layer.Dense(1)
    ])
    loss = tn.loss.MSE()
    optimizer = tn.optimizer.Adam(0.01)
    return tn.model.Model(net=net, loss=loss, optimizer=optimizer)


def test_model_save_and_load(model, tmpdir):
    model_copy = copy.deepcopy(model)
    # simple training
    x = np.random.uniform(0.0, 1.0, (100, 3))
    y = np.random.uniform(0.0, 1.0, (100, 1))

    for _ in range(5):
        pred = model.forward(x)
        _, grads = model.backward(pred, y)
        model.apply_grads(grads)

    pred = model.forward(x)
    mse_save, _ = tn.metric.mean_square_error(pred, y)
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
    mse_load, _ = tn.metric.mean_square_error(pred, y)
    assert mse_save == mse_load
