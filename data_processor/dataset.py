# Author: borgwang <borgwang@126.com>
# Date: 2018-05-23
#
# Filename: dataset.py
# Description: dataset class


import gzip
import os
import pickle
from urllib.error import URLError
from urllib.request import urlretrieve


class Dataset(object):

    def __init__(self, dir, transform=None):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir = dir
        self.transform = transform


class MNISTDataset(Dataset):

    def __init__(self, dir, transform=None):
        super().__init__(dir, transform)
        url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
        path = os.path.join(self.dir, url.split("/")[-1])
        self._download(path, url)
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

    @staticmethod
    def _download(path, url):
        try:
            if os.path.exists(path):
                print("{} already exists.".format(path))
            else:
                print("Downloading {}.".format(url))
                try:
                    urlretrieve(url, path)
                except URLError:
                    raise RuntimeError("Error downloading resource!")
                finally:
                    print()
        except KeyboardInterrupt:
            print("Interrupted")

    def _load(self, path):
        print("Loading MNIST dataset.")
        with gzip.open(path, "rb") as f:
            self._train_set, self._valid_set, self._test_set = \
                pickle.load(f, encoding="latin1")
