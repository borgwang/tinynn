"""Evaluator class."""

import numpy as np


class BaseEvaluator(object):

    @classmethod
    def evaluate(cls, predictions, targets):
        raise NotImplementedError("Must specify evaluator.")


class AccEvaluator(BaseEvaluator):

    @classmethod
    def evaluate(cls, predictions, targets):
        total_num = len(predictions)
        hit_num = int(np.sum(predictions == targets))
        result = {"total_num": total_num,
                  "hit_num": hit_num,
                  "accuracy": 1.0 * hit_num / total_num}

        return result


class PrecisionEvaluator(BaseEvaluator):

    @classmethod
    def evaluate(cls, predictions, targets):
        pass


class RecallEvaluator(BaseEvaluator):
    @classmethod
    def evaluate(cls, predictions, targets):
        pass


class F1Evaluator(BaseEvaluator):

    @classmethod
    def evaluate(cls, predictions, targets):
        pass


class ROCEvaluator(BaseEvaluator):

    @classmethod
    def evaluate(cls, predictions, targets):
        pass


class EVEvaluator(BaseEvaluator):
    """
    Explained variance evaluator computes fraction of variance that pred_y explains about y.
    Returns 1 - Var[y-pred_y] / Var[y]

    Interpretation:
        EV=0  =>  might as well have predicted zero
        EV=1  =>  perfect prediction
        EV<0  =>  worse than just predicting zero
    """
    @classmethod
    def evaluate(cls, predictions, targets):
        assert predictions.shape == targets.shape
        if predictions.ndim == 1:
            diff_var = np.var(targets - predictions)
            target_var = np.var(targets)
        elif predictions.ndim == 2:
            diff_var = np.var(targets - predictions, axis=0)
            target_var = np.var(targets, axis=0)

        non_zero_idx = np.where(target_var != 0)[0]

        ev = np.mean(1.0 - diff_var[non_zero_idx] / target_var[non_zero_idx])
        res = {"mean_ev": ev}
        return res


class MSEEvaluator(BaseEvaluator):
    """ Mean square error evaluator"""
    @classmethod
    def evaluate(cls, predictions, targets):
        assert predictions.shape == targets.shape
        if predictions.ndim == 1:
            mse = np.mean(np.square(predictions - targets))
        elif predictions.ndim == 2:
            mse = np.mean(np.sum(np.square(predictions - targets), axis=1))
        else:
            raise ValueError("predision supposes to have 1 or 2 dim.")
        res = {"mse": mse}
        return res


class MAEEvaluator(BaseEvaluator):
    """ Mean absolute error evaluator"""
    @classmethod
    def evaluate(cls, predictions, targets):
        assert predictions.shape == targets.shape
        if predictions.ndim == 1:
            mse = np.mean(np.abs(predictions - targets))
        elif predictions.ndim == 2:
            mse = np.mean(np.sum(np.abs(predictions - targets), axis=1))
        else:
            raise ValueError("predision supposes to have 1 or 2 dim.")
        res = {"mse": mse}
        return res


class R2Evaluator(BaseEvaluator):
    """ R-square Evaluator"""
    @classmethod
    def evaluate(cls, predictions, targets):
        pass
