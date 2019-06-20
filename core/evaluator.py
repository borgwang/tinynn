# Author: borgwang <borgwang@126.com>
# Date: 2018-05-29
#
# Filename: evaluator.py
# Description: model evaluating class


import numpy as np


class BaseEvaluator(object):

    @classmethod
    def eval(cls, preds, targets):
        '''
        "preds" and "targets" must be numpy arrays.
        '''
        raise NotImplementedError("Must specipy evaluator.")


# ----------
# Classification evaluators
# ----------

class AccEvaluator(BaseEvaluator):

    @classmethod
    def eval(cls, preds, targets):
        assert len(preds) == len(targets)
        total_num = len(preds)
        hit_num = int((preds == targets).sum())
        result = {"total_num": total_num,
                  "hit_num": hit_num,
                  "accuracy": 1.0 * hit_num / total_num}

        return result


class PrecisionEvaluator(BaseEvaluator):
    pass


class RecallEvaluator(BaseEvaluator):
    pass


class F1Evaluator(BaseEvaluator):
    pass


class ROCEvaluator(BaseEvaluator):
    pass


# ----------
# Regression evaluators
# ----------

class EVEvaluator(BaseEvaluator):
    '''
    Explained variance evaluator computes fraction of variance that pred_y explains about y.
    Returns 1 - Var[y-pred_y] / Var[y]

    Interpretation:
        EV=0  =>  might as well have predicted zero
        EV=1  =>  perfect prediction
        EV<0  =>  worse than just predicting zero
    '''
    @classmethod
    def eval(cls, preds, targets):
        assert preds.shape == targets.shape
        if preds.ndim == 1:
            diff_var = np.var(targets - preds)
            target_var = np.var(targets)
        elif preds.ndim == 2:
            diff_var = np.var(targets - preds, axis=0)
            target_var = np.var(targets, axis=0)

        non_zero_idx = np.where(target_var != 0)[0]

        ev = np.mean(1.0 - diff_var[non_zero_idx] / target_var[non_zero_idx])
        res = {"mean_ev": ev}
        return res


class MSEEvaluator(BaseEvaluator):
    ''' Mean square error evaluator'''
    @classmethod
    def eval(cls, preds, targets):
        assert preds.shape == targets.shape
        if preds.ndim == 1:
            mse = np.mean(np.square(preds - targets))
        elif preds.ndim == 2:
            mse = np.mean(np.sum(np.square(preds - targets), axis=1))
        else:
            raise ValueError("predision supposes to have 1 or 2 dim.")
        res = {"mse": mse}
        return res


class MAEEvaluator(BaseEvaluator):
    ''' Mean absolute error evaluator'''
    @classmethod
    def eval(cls, preds, targets):
        assert preds.shape == targets.shape
        if preds.ndim == 1:
            mse = np.mean(np.abs(preds - targets))
        elif preds.ndim == 2:
            mse = np.mean(np.sum(np.abs(preds - targets), axis=1))
        else:
            raise ValueError("predision supposes to have 1 or 2 dim.")
        res = {"mse": mse}
        return res


class R2Evaluator(BaseEvaluator):
    ''' R-square Evalutor'''
    pass
