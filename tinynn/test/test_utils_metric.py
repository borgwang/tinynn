import runtime_path  # isort:skip

import numpy as np

from tinynn.utils.metric import *


def approximately_equal(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_classification_metrics():
    scores = np.array([.1, .2, .6, .7, .8, .9, .8, .7, .3, .4])
    predictions = (scores >= 0.5).astype(int)
    targets = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1])

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
    assert approximately_equal(auc_score, 0.45833)
    auc_score2 = auc(scores, targets)["auc"]
    assert approximately_equal(auc_score, 0.45833)

    # log loss
    log_loss_score = log_loss(scores, targets)["log_loss"]
    assert approximately_equal(log_loss_score, 0.87889)


def test_regression_metrics():
    predictions = np.array([1., 2.1, 3., 4.1, 5.])
    targets = np.array([1.1, 2., 3.1, 4., 5.1])
    # mse
    mse = mean_square_error(predictions, targets)["mse"]
    assert approximately_equal(mse, 0.01)
    # mae
    mae = mean_absolute_error(predictions, targets)["mae"]
    assert approximately_equal(mae, 0.1)
    # r2
    r2 = r_square(predictions, targets)["r_square"]
    assert approximately_equal(r2, 0.995)
    # explained_variation
    ev = explained_variation(predictions, targets)["mean_ev"]
    assert approximately_equal(ev, 0.995)
