"""Methods to compute common machine learning metrics"""

import numpy as np


def accuracy(predictions, targets):
    total_num = len(predictions)
    hit_num = int(np.sum(predictions == targets))
    return {"total_num": total_num,
            "hit_num": hit_num,
            "accuracy": 1.0 * hit_num / total_num}


def log_loss(predictions, targets):
    assert len(predictions) == len(targets)
    total_num = len(predictions)

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    total_log = np.log(predictions[range(total_num), targets])
    logloss = 1.0 - np.mean(total_log)
    return {"log_loss": logloss}


def precision(predictions, targets, pos_class=1, neg_class=0):
    """precision = TP / (TP + FP)"""
    true_pos, false_pos = 0, 0
    for pred, tar in zip(predictions, targets):
        if pred == pos_class and tar == pos_class:
            true_pos += 1.0
        elif pred == pos_class and tar == neg_class:
            false_pos += 1.0
    precision = true_pos / (true_pos + false_pos)
    return {"precision": precision, "true_positive": true_pos, 
            "false_positive": false_pos}


def recall(predictions, targets, pos_class=1, neg_class=0):
    """recall = TP / (TP + FN)"""
    true_pos, false_neg = 0, 0
    for pred, tar in zip(predictions, targets):
        if pred == pos_class and tar == pos_class:
            true_pos += 1.0
        elif pred == neg_class and tar == pos_class:
            false_neg += 1.0
    recall = true_pos / (true_pos + false_neg)
    return {"recall": recall, "true_positive": true_pos,
            "false_negative": false_neg}


def f1(predictions, targets, pos_class=1, neg_class=0):
    p = precision(predictions, targets, pos_class, neg_class)
    r = recall(predictions, targets, pos_class, neg_class)
    return {"f1": 2 * (p * r) / (p + r), "precision": p, "recall": r}


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
    elif predictions.ndim == 2:
        diff_var = np.var(targets - predictions, axis=0)
        target_var = np.var(targets, axis=0)

    non_zero_idx = np.where(target_var != 0)[0]

    ev = np.mean(1.0 - diff_var[non_zero_idx] / target_var[non_zero_idx])
    return {"mean_ev": ev}


def r_square(predictions, targets):
    assert predictions.shape == targets.shape
    ss_residual = np.sum(targets - predictions, axis=0)
    ss_total = np.sum(targets - np.mean(targets, axis=0), axis=0)
    r2 = np.mean(1 - ss_residual / ss_total)
    return {"r_square": r2}


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
