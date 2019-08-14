# Author: borgwang <borgwang@126.com>
# Date: 2019-06-20
#
# Filename: test_utils_seeder.py
# Description: test unit for utils/seeder.py


import runtime_path  # isort:skip

import pytest
from utils.seeder import random_seed


def test_random_seed():
    with pytest.raises(ValueError):
        random_seed(2**32 + 1)
