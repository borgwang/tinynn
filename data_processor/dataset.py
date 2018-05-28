import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import urllib
import pickle
import gzip
from urllib.error import URLError
from urllib.request import urlretrieve


class Dataset(object):

    def __init__(self, dir, transform=None):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir: str = dir


class MNIST(Dataset):

    def __init__(self, dir, transform=None):
        super().__init__(dir, transform)
        URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        path = os.path.join(self.dir, URL.split('/')[-1])
        self._download(path, URL)
        self._load(path)

    @property
    def train_data(self):
        return self._train_set

    @property
    def test_data(self):
        return self._test_set

    @property
    def valid_data(self):
        return self._valid_set

    def _download(self, path, url):
        try:
            if os.path.exists(path):
                print('{} already exists, skipping...'.format(path))
            else:
                print('Downloading {}...'.format(url))
                try:
                    urlretrieve(url, path)
                except URLError:
                    raise RuntimeError('Error downloading resource!')
                finally:
                    print()
        except KeyboardInterrupt:
            print('Interrupted')

    def _load(self, path):
        print('Loading MNIST dataset...')
        with gzip.open(path, 'rb') as f:
            self._train_set, self._valid_set, self._test_set = pickle.load(f, encoding='latin1')
