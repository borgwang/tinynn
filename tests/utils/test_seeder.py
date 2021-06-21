import numpy as np
import pytest
import tinynn as tn



def test_random_seed():
    with pytest.raises(ValueError):
        tn.seeder.random_seed(2 ** 32 + 1)
    tn.seeder.random_seed(0)
    assert (np.random.randint(0, 10, 3) == [5, 0, 3]).all()
