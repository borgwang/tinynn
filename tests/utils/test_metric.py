import numpy as np
import tinynn as tn


def test_classification_metrics():
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.3, 0.4])
    scores = scores.reshape((-1, 1))
    predictions = (scores >= 0.5).astype(int)
    targets = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1]).reshape((-1, 1))

    # precision
    precision, info = tn.metric.precision(predictions, targets)
    assert precision == 0.5
    assert info["true_positive"] == 3
    assert info["false_positive"] == 3
    # recall
    recall, info = tn.metric.recall(predictions, targets)
    assert recall == 0.5
    assert info["true_positive"] == 3
    assert info["false_negative"] == 3
    # f1
    assert tn.metric.f1_score(predictions, targets)[0] == 0.5
    # accuracy
    assert tn.metric.accuracy(predictions, targets)[0] == 0.4
    # auc
    auc_score = tn.metric.auc_roc_curve(scores, targets)[0]
    assert np.allclose(auc_score, 0.45833)
    auc_score2 = tn.metric.auc(scores, targets)[0]
    assert np.allclose(auc_score2, 0.45833)

    # log loss
    log_loss_score = tn.metric.log_loss(scores, targets)[0]
    assert np.allclose(log_loss_score, 0.87889)


def test_regression_metrics():
    predictions = np.array([1.0, 2.1, 3.0, 4.1, 5.]).reshape((-1, 1))
    targets = np.array([1.1, 2.0, 3.1, 4.0, 5.1]).reshape((-1, 1))
    # mse
    mse = tn.metric.mean_square_error(predictions, targets)[0]
    assert np.allclose(mse, 0.01)
    # mae
    mae = tn.metric.mean_absolute_error(predictions, targets)[0]
    assert np.allclose(mae, 0.1)
    # r2
    r2 = tn.metric.r_square(predictions, targets)[0]
    assert np.allclose(r2, 0.9950)
    # explained_variation
    ev = tn.metric.explained_variation(predictions, targets)[0]
    assert np.allclose(ev, 0.9952)
