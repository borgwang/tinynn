"""Common datasets"""

import gzip
import os
import pickle
import struct
import tarfile

import numpy as np

from tinynn.utils.downloader import download_url


class Dataset:

    def __init__(self, data_dir, **kwargs):
        self._train_set = None
        self._valid_set = None
        self._test_set = None

        self._save_paths = [os.path.join(data_dir, url.split("/")[-1])
                            for url in self._urls]

        self._download()
        self._parse(**kwargs)  # lgtm [py/init-calls-subclass]

    def _download(self):
        for url, checksum, save_path in zip(
                self._urls, self._checksums, self._save_paths):
            download_url(url, save_path, checksum)

    def _parse(self, **kwargs):
        raise NotImplementedError

    @property
    def train_set(self):
        return self._train_set

    @property
    def valid_set(self):
        return self._valid_set

    @property
    def test_set(self):
        return self._test_set

    @staticmethod
    def get_one_hot(targets, n_classes):
        return np.eye(n_classes)[np.array(targets).reshape(-1)]


class MNIST(Dataset):

    def __init__(self, data_dir, one_hot=True):
        self._urls = ("https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz",)
        self._checksums = ("98100ca27dc0e07ddd9f822cf9d244db",)
        self._n_classes = 10
        super().__init__(data_dir, one_hot=one_hot)

    def _parse(self, **kwargs):
        save_path = self._save_paths[0]
        with gzip.open(save_path, "rb") as f:
            train, valid, test = pickle.load(f, encoding="latin1")

        if kwargs["one_hot"]:
            train = (train[0], self.get_one_hot(train[1], self._n_classes))
            valid = (valid[0], self.get_one_hot(valid[1], self._n_classes))
            test = (test[0], self.get_one_hot(test[1], self._n_classes))

        self._train_set, self._valid_set, self._test_set = train, valid, test


class FashionMNIST(Dataset):

    def __init__(self, data_dir, one_hot=True):
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        self._urls = [base_url + "train-images-idx3-ubyte.gz",
                      base_url + "train-labels-idx1-ubyte.gz",
                      base_url + "t10k-images-idx3-ubyte.gz",
                      base_url + "t10k-labels-idx1-ubyte.gz"]
        self._checksums = ["8d4fb7e6c68d591d4c3dfef9ec88bf0d",
                           "25c81989df183df01b3e8a0aad5dffbe",
                           "bef4ecab320f06d8554ea6380940ec79",
                           "bb300cfdad3c16e7a12a480ee83cd310"]
        self._n_classes = 10
        super().__init__(data_dir, one_hot=one_hot)

    @staticmethod
    def read_idx(filename):
        with gzip.open(filename, "rb") as fileobj:
            _, _, dims = struct.unpack(">HBB", fileobj.read(4))
            shape = tuple(struct.unpack(">I", fileobj.read(4))[0]
                          for d in range(dims))
            return np.frombuffer(fileobj.read(), dtype=np.uint8).reshape(shape)

    def _parse(self, **kwargs):
        train_x, train_y, test_x, test_y = (self.read_idx(path)
                                            for path in self._save_paths)
        # normalize
        train_x = train_x.astype(float) / 255.0
        test_x = test_x.astype(float) / 255.0
        train_x = train_x.reshape((len(train_x), -1))
        test_x = test_x.reshape((len(test_x), -1))

        if kwargs["one_hot"]:
            train_y = self.get_one_hot(train_y, self._n_classes)
            test_y = self.get_one_hot(test_y, self._n_classes)
        self._train_set = (train_x, train_y)
        self._test_set = (test_x, test_y)


class Cifar(Dataset):

    @staticmethod
    def _cifar_normalize(data):
        means, stds = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        data = data.reshape((len(data), -1, 3))
        for ch in range(3):
            data[:, :, ch] = (data[:, :, ch] - means[ch]) / stds[ch]
        data = data.reshape((len(data), -1))
        return data

    def _parse(self, **kwargs):
        raise NotImplementedError

    def _parse_tarfile(self):
        dataset = {}
        with open(self._save_paths[0], "rb") as f:
            tar = tarfile.open(fileobj=f)
            for item in tar:
                obj = tar.extractfile(item)
                if not obj or item.size < 100:
                    continue
                cont = pickle.load(obj, encoding="bytes")
                dataset[item.name.split("/")[-1]] = cont
        return dataset


class Cifar10(Cifar):

    def __init__(self, data_dir, one_hot=False, normalize=False):
        self._urls = ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",)
        self._checksums = ("c58f30108f718f92721af3b95e74349a",)
        self._n_classes = 10
        super().__init__(data_dir, one_hot=one_hot, normalize=normalize)

    def _parse(self, **kwargs):
        dataset = self._parse_tarfile()
        data_batch_name = ["data_batch_%d" % i for i in range(1, 6)]
        train_x, train_y = [], []
        for name in data_batch_name:
            cont = dataset[name]
            train_x.append(cont[b"data"])
            train_y.extend(cont[b"labels"])
        train_x = np.concatenate(train_x, axis=0)

        train_x = train_x / 255.
        train_y = np.asarray(train_y)
        test_x = dataset["test_batch"][b"data"] / 255.
        test_y = np.asarray(dataset["test_batch"][b"labels"])

        if kwargs["normalize"]:
            train_x = self._cifar_normalize(train_x)
            test_x = self._cifar_normalize(test_x)

        if kwargs["one_hot"]:
            train_y = self.get_one_hot(train_y, self._n_classes)
            test_y = self.get_one_hot(test_y, self._n_classes)

        self._train_set = (train_x, train_y)
        self._test_set = (test_x, test_y)


class Cifar100(Cifar):

    def __init__(self, data_dir, one_hot=False, normalize=False):
        self._urls = ("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",)
        self._checksums = ("eb9058c3a382ffc7106e4002c42a8d85",)
        self._n_classes = 100
        super().__init__(data_dir, one_hot=one_hot, normalize=normalize)

    def _parse(self, **kwargs):
        dataset = self._parse_tarfile()
        train_x = dataset["train"][b"data"] / 255.
        train_y = np.asarray(dataset["train"][b"fine_labels"])

        test_x = dataset["test"][b"data"] / 255.
        test_y = np.asarray(dataset["test"][b"fine_labels"])

        if kwargs["normalize"]:
            train_x = self._cifar_normalize(train_x)
            test_x = self._cifar_normalize(test_x)

        if kwargs["one_hot"]:
            train_y = self.get_one_hot(train_y, self._n_classes)
            test_y = self.get_one_hot(test_y, self._n_classes)

        self._train_set = (train_x, train_y)
        self._test_set = (test_x, test_y)
