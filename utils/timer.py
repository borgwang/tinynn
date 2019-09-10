"""A utility for timing."""

import time

import numpy as np


class Timer(object):

    def __init__(self, task_name="UntitledTask"):
        self.task_name = task_name
        self._duration_list = []
        self.now = None
        self.check_point = None
        self.is_timing = False
        self._count = 0

    def start(self):
        if not self.is_timing:
            self.check_point = time.time()
            self.is_timing = True

    def pause(self):
        if self.is_timing:
            self._duration_list.append(time.time() - self.check_point)
            self.is_timing = False
            self._count += 1

    def stop(self):
        self.pause()
        self.report()

    def report(self):
        print("[Timer] {} total: {:.4f} mean: {:.4f} count: {}".format(
            self.task_name, np.sum(self._duration_list),
            np.mean(self._duration_list), self._count))

    @property
    def duration(self):
        return np.sum(self._duration_list)

    @property
    def count(self):
        return self._count
