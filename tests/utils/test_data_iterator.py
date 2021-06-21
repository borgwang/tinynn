import numpy as np
import tinynn as tn


def test_batch_iterator():
    batch_size = 10
    n_data = 10 * batch_size  # 10 batches

    iterator = tn.data_iterator.BatchIterator(batch_size=batch_size)
    x_dim, y_dim = 10, 5
    fake_x = np.random.randint(0, 100, size=(n_data, x_dim))
    fake_y = np.random.randint(0, 100, size=(n_data, y_dim))

    n_batches = 0
    for batch_x, batch_y in iterator(fake_x, fake_y):
        assert batch_x.shape == (batch_size, x_dim)
        assert batch_y.shape == (batch_size, y_dim)
        n_batches += 1

    assert n_batches == 10
