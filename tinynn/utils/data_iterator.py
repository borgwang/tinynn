"""Data Iterator class."""

from collections import namedtuple

import numpy as np

Batch = namedtuple("Batch", ["inputs", "targets"])


class BaseIterator:

    def __call__(self, inputs, targets):
        raise NotImplementedError


class BatchIterator(BaseIterator):

    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        indices = np.arange(len(inputs))
        if self.shuffle:
            np.random.shuffle(indices)

        starts = np.arange(0, len(inputs), self.batch_size)
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[indices[start: end]]
            batch_targets = targets[indices[start: end]]
            yield Batch(inputs=batch_inputs, targets=batch_targets)
