import numpy as np
import pytest

from tinynn.utils.seeder import random_seed


def test_random_seed():
    with pytest.raises(ValueError):
        random_seed(2 ** 32 + 1)
    random_seed(0)
    assert (np.random.randint(0, 10, 3) == [5, 0, 3]).all()
