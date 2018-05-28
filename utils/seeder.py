# Author: borgwang <borgwang@126.com>
# Date: 2018-05-28
#
# Filename: seeder.py
# Description: random seed


import numpy as np


def random_seed(seed):
    seed = int(seed)
    if seed < 0 or seed > (2**32 - 1):
        raise ValueError('Seed must be between 0 and 2**32 - 1')
    else:
        np.random.seed(seed)
