import numpy as np

from tinynn.utils.metric import *


def test_classification_metrics():
    scores = np.array([.1, .2, .6, .7, .8, .9, .8, .7, .3, .4]).reshape((-1, 1))
    predictions = (scores >= 0.5).astype(int)
    targets = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1]).reshape((-1, 1))

    # precision
    result = precision(predictions, targets)
    assert result["true_positive"] == 3
    assert result["false_positive"] == 3
    assert result["precision"] == 0.5
    # recall
    result = recall(predictions, targets)
    assert result["true_positive"] == 3
    assert result["false_negative"] == 3
    assert result["recall"] == 0.5
    # f1
    assert f1_score(predictions, targets)["f1"] == 0.5
    # accuracy
    assert accuracy(predictions, targets)["accuracy"] == 0.4
    # auc
    auc_score = auc_roc_curve(scores, targets)["auc"]
    assert np.allclose(auc_score, 0.45833)
    auc_score2 = auc(scores, targets)["auc"]
    assert np.allclose(auc_score2, 0.45833)

    # log loss
    log_loss_score = log_loss(scores, targets)["log_loss"]
    assert np.allclose(log_loss_score, 0.87889)


def test_regression_metrics():
    predictions = np.array([1., 2.1, 3., 4.1, 5.]).reshape((-1, 1))
    targets = np.array([1.1, 2., 3.1, 4., 5.1]).reshape((-1, 1))
    # mse
    mse = mean_square_error(predictions, targets)["mse"]
    assert np.allclose(mse, 0.01)
    # mae
    mae = mean_absolute_error(predictions, targets)["mae"]
    assert np.allclose(mae, 0.1)
    # r2
    r2 = r_square(predictions, targets)["r_square"]
    assert np.allclose(r2, 0.9950)
    # explained_variation
    ev = explained_variation(predictions, targets)["mean_ev"]
    assert np.allclose(ev, 0.9952)
