"""Methods to compute common machine learning metrics"""

import numpy as np


def _roc_curve(predictions, targets, partition, pos_class, neg_class):
    """ROC curve (for binary classification only)"""
    thresholds = np.arange(0.0, 1.0, 1.0 / partition)[::-1]
    fpr_list, tpr_list = [], []
    for threshold in thresholds:
        pred_class = (predictions >= threshold).astype(int)
        false_pos = np.sum((pred_class == pos_class) & (targets == neg_class))
        fpr_list.append(false_pos / np.sum(targets == neg_class))
        true_pos = np.sum((pred_class == pos_class) & (targets == pos_class))
        tpr_list.append(true_pos / np.sum(targets == pos_class))
    return fpr_list, tpr_list, thresholds


def auc_roc_curve(predictions, targets, partition=300, pos_class=1, neg_class=0):
    """Area unser the ROC curve (for binary classification only)"""
    fprs, tprs, thresholds = _roc_curve(predictions, targets, partition, pos_class, neg_class)
    auc_ = 0.0
    for i in range(len(thresholds) - 1):
        auc_ += tprs[i] * (fprs[i + 1] - fprs[i])
    return {"auc": auc_, "fprs": fprs, "tprs": tprs, "thresholds": thresholds}


def auc(predictions, targets, pos_class=1, neg_class=0):
    num_pos = np.sum(targets == pos_class)
    num_neg = np.sum(targets == neg_class)
    idx = np.argsort(predictions)
    sorted_targets = targets[idx]
    cnt = 0
    for i, target in enumerate(sorted_targets):
        if target == pos_class:
            cnt += np.sum(sorted_targets[:i] == neg_class)
    auc_ = 1. * cnt / (num_pos * num_neg)
    return {"auc": auc_}


def accuracy(predictions, targets):
    total_num = len(predictions)
    hit_num = int(np.sum(predictions == targets))
    return {"total_num": total_num,
            "hit_num": hit_num,
            "accuracy": 1.0 * hit_num / total_num}


def log_loss(predictions, targets):
    assert len(predictions) == len(targets)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    logloss = np.mean(-(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)))
    return {"log_loss": logloss}


def precision(predictions, targets, pos_class=1, neg_class=0):
    """precision = TP / (TP + FP)"""
    assert len(predictions) == len(targets)
    true_pos = np.sum((predictions == pos_class) & (targets == pos_class))
    false_pos = np.sum((predictions == pos_class) & (targets == neg_class))
    precision_ = 1. * true_pos / (true_pos + false_pos)
    return {"precision": precision_, "true_positive": true_pos, "false_positive": false_pos}


def recall(predictions, targets, pos_class=1, neg_class=0):
    """recall = TP / (TP + FN)"""
    assert len(predictions) == len(targets)
    true_pos = np.sum((predictions == pos_class) & (targets == pos_class))
    false_neg = np.sum((predictions == neg_class) & (targets == pos_class))
    recall_ = 1. * true_pos / (true_pos + false_neg)
    return {"recall": recall_, "true_positive": true_pos, "false_negative": false_neg}


def f1_score(predictions, targets, pos_class=1, neg_class=0):
    precision_ = precision(predictions, targets, pos_class, neg_class)["precision"]
    recall_ = recall(predictions, targets, pos_class, neg_class)["recall"]
    return {"f1": 2 * (precision_ * recall_) / (precision_ + recall_),
            "precision": precision_, "recall": recall_}


def explained_variation(predictions, targets):
    """
    Computes fraction of variance that pred_y explains about y.
    Returns 1 - Var[y-pred_y] / Var[y]

    Interpretation:
        EV=0  =>  might as well have predicted zero
        EV=1  =>  perfect prediction
        EV<0  =>  worse than just predicting zero
    """
    assert predictions.shape == targets.shape
    if predictions.ndim == 1:
        diff_var = np.var(targets - predictions)
        target_var = np.var(targets)
        ev_ = 1.0 - diff_var / target_var
    elif predictions.ndim == 2:
        diff_var = np.var(targets - predictions, axis=0)
        target_var = np.var(targets, axis=0)
        non_zero_idx = np.where(target_var != 0)[0]
        ev_ = np.mean(1.0 - diff_var[non_zero_idx] / target_var[non_zero_idx])
    return {"mean_ev": ev_}


def r_square(predictions, targets):
    assert predictions.shape == targets.shape
    ss_residual = np.sum((targets - predictions) ** 2, axis=0)
    ss_total = np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=0)
    r_square_ = np.mean(1 - ss_residual / ss_total)
    return {"r_square": r_square_}


def mean_square_error(predictions, targets):
    assert predictions.shape == targets.shape
    if predictions.ndim == 1:
        mse = np.mean(np.square(predictions - targets))
    elif predictions.ndim == 2:
        mse = np.mean(np.sum(np.square(predictions - targets), axis=1))
    else:
        raise ValueError("predictions supposes to have 1 or 2 dim.")
    return {"mse": mse}


def mean_absolute_error(predictions, targets):
    assert predictions.shape == targets.shape
    if predictions.ndim == 1:
        mae = np.mean(np.abs(predictions - targets))
    elif predictions.ndim == 2:
        mae = np.mean(np.sum(np.abs(predictions - targets), axis=1))
    else:
        raise ValueError("predictions supposes to have 1 or 2 dim.")
    return {"mae": mae}
