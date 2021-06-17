import pytest
from tinynn.utils.seeder import random_seed


def test_random_seed():
    with pytest.raises(ValueError):
        random_seed(2 ** 32 + 1)
