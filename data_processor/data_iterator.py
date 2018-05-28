# Author: borgwang <borgwang@126.com>
# Date: 2018-05-23
#
# Filename: data_iterator.py
# Description: Data Iterator class


from typing import Iterator, NamedTuple
import numpy as np

from core.tensor import Tensor


Batch = NamedTuple('Batch', [('inputs', Tensor), ('targets', Tensor)])


class BaseIterator(object):

    def __call__(self, inputs, targets):
        raise NotImplementedError


class BatchIterator(BaseIterator):

    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            idx = np.arange(len(inputs))
            np.random.shuffle(idx)
            inputs = inputs[idx]
            targets = targets[idx]

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start: end]
            batch_targets = targets[start: end]
            yield Batch(batch_inputs, batch_targets)
