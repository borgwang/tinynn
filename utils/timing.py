# Author: borgwang <borgwang@126.com>
# Date: 2018-05-28
#
# Filename: timing.py
# Description: timer


import time
import numpy as np


def timing(fn):
    def wrapped_fn(*args, **kwargs):
        t_start = time.time()
        ret = fn(*args, **kwargs)
        print('%.3fs taken for %s' % (time.time() - t_start, fn.__name__))
        return ret
    return wrapped_fn


class Timer(object):

    def __init__(self, task_name):
        self.task_name = task_name
        self.duration = []
        self.now = None
        self.is_timing = False
        self.count = 0

    def start(self):
        if not self.is_timing:
            self.check_point = time.time()
            self.is_timing = True

    def pause(self):
        if self.is_timing:
            self.duration.append(time.time() - self.check_point)
            self.is_timing = False
            self.count += 1

    def stop(self):
        self.pause()
        self.report()

    def report(self):
        print('--- Timer: {} total: {:.4f} mean: {:.4f} count: {} ---'.format(
            self.task_name, np.sum(self.duration), np.mean(self.duration), self.count))
