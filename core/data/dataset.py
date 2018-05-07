from typing import Tuple

import os
import numpy as np
import urllib
import pickle
import gzip
try:
    from urllib.error import URLError
    from urllib.request import urlretrieve
except ImportError:
    from urllib2 import URLError
    from urllib2 import urlretrieve

from core.tensor import Tensor
from core.data.transform import Transform


class Dataset(object):
    pass


class MNIST(Dataset):

    def __init__(self,
                 dir: str,
                 transform: Transform = None) -> None:
        URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, URL.split('/')[-1])
        self._download(path, URL)
        self._load(path)

    def get_train_data(self) -> Tuple[Tensor, Tensor]:
        return self._train_set

    def get_test_data(self) -> Tuple[Tensor, Tensor]:
        return self._test_set

    def get_valid_data(self) -> Tuple[Tensor, Tensor]:
        return self._valid_set

    def _download(self, path: str, url: str) -> None:
        try:
            if os.path.exists(path):
                print('{} already exists, skipping ...'.format(path))
            else:
                print('Downloading {} ...'.format(url))
                try:
                    urlretrieve(url, path)
                except URLError:
                    raise RuntimeError('Error downloading resource!')
                finally:
                    print()
        except KeyboardInterrupt:
            print('Interrupted')

    def _load(self, path: str) -> None:
        print('Loading MNIST dataset...')
        with gzip.open(path, 'rb') as f:
            self._train_set, self._valid_set, self._test_set = pickle.load(f, encoding='latin1')
