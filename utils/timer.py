# Author: borgwang <borgwang@126.com>
# Date: 2018-05-28
#
# Filename: timer.py
# Description: timer function


import time


def Timer(fn):
    def wrapped_fn(*args, **kwargs):
        t_start = time.time()
        ret = fn(*args, **kwargs)
        print('%.3fs taken for %s' % (time.time() - t_start, fn.__name__))
        return ret
    return wrapped_fn
