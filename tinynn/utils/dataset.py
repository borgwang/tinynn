"""Common datasets"""

import gzip
import os
import pickle
import sys
import struct
import tarfile

import numpy as np

from tinynn.utils.downloader import download_url


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def mnist(data_dir, one_hot=False):
    """
    return: train_set, valid_set, test_set
    train_set size: (50000, 784), (50000,)
    valid_set size: (10000, 784), (10000,)
    test_set size: (10000, 784), (10000,)
    feature: numerical in range [0, 1]
    target: categorical from 0 to 9
    """
    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    checksum = "a02cd19f81d51c426d7ca14024243ce9"

    save_path = os.path.join(data_dir, url.split("/")[-1])
    print("Preparing MNIST dataset ...")
    try:
        download_url(url, save_path, checksum)
    except Exception as e:
        print("Error downloading dataset: %s" % str(e))
        sys.exit(1)

    # load the dataset
    with gzip.open(save_path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        valid_set = (valid_set[0], get_one_hot(valid_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))

    return train_set, valid_set, test_set


def fashion_mnist(data_dir, one_hot=False):
    """
    return: train_set, test_set
    train_set size: (60000, 784), (60000,)
    test_set size: (10000, 784), (10000,)
    feature: numerical in range [0, 1]
    target: categorical from 0 to 9
    """
    def read_idx(filename):
        with gzip.open(filename, "rb") as f:
            zero, data_type, dims = struct.unpack(">HBB", f.read(4))
            shape = tuple(struct.unpack(">I", f.read(4))[0] for d in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

    urls = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"]
    checksums = ["8d4fb7e6c68d591d4c3dfef9ec88bf0d",
                 "25c81989df183df01b3e8a0aad5dffbe",
                 "bef4ecab320f06d8554ea6380940ec79",
                 "bb300cfdad3c16e7a12a480ee83cd310"]
    save_paths = [os.path.join(data_dir, url.split("/")[-1]) for url in urls]

    print("Preparing Fashion-MNIST dataset...")
    for url, checksum, save_path in zip(urls, checksums, save_paths):
        try:
            download_url(url, save_path, checksum)
        except Exception as e:
            print("Error downloading dataset: %s" % str(e))
            sys.exit(1)

    dataset = []
    for path in save_paths:
        dataset.append(read_idx(path))
    train_x, train_y, test_x, test_y = dataset
    # normalize
    train_x = train_x.astype(float) / 255.0
    test_x = test_x.astype(float) / 255.0
    # reshape
    train_x = train_x.reshape((len(train_x), -1))
    test_x = test_x.reshape((len(test_x), -1))
    # one hot if need
    if one_hot:
        train_y = get_one_hot(train_y, 10)
        test_y = get_one_hot(test_y, 10)
    return (train_x, train_y), None, (test_x, test_y)


def cifar10(data_dir, one_hot=False):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    checksum = "c58f30108f718f92721af3b95e74349a"

    save_path = os.path.join(data_dir, url.split("/")[-1])
    print("Preparing CIFAR-10 dataset...")
    try:
        download_url(url, save_path, checksum)
    except Exception as e:
        print("Error downloading dataset: %s" % str(e))
        sys.exit(1)

    # load the dataset
    dataset = {}
    with open(save_path, "rb") as f:
        tar = tarfile.open(fileobj=f)
        for item in tar:
            obj = tar.extractfile(item)
            if not obj or item.size < 100:
                continue
            cont = pickle.load(obj, encoding="bytes")
            dataset[item.name.split("/")[-1]] = cont

    data_batch_name = ["data_batch_%d" % i for i in range(1, 6)]
    train_x, train_y = [], []
    for name in data_batch_name:
        cont = dataset[name]
        train_x.append(cont[b'data'])
        train_y.extend(cont[b'labels'])
    train_x = np.concatenate(train_x, axis=0)

    # normalize
    means, stds = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_x = train_x / 255.0
    train_x = train_x.reshape((-1, 1024, 3))
    for c in range(3):
        train_x[:, :, c] = (train_x[:, :, c] - means[c]) / stds[c]
    train_x = train_x.reshape(-1, 3072)

    train_y = np.asarray(train_y)
    train_set = (train_x, train_y)

    test_x = dataset["test_batch"][b"data"]
    test_x = test_x / 255.0
    test_x = test_x.reshape((-1, 1024, 3))
    for c in range(3):
        test_x[:, :, c] = (test_x[:, :, c] - means[c]) / stds[c]
    test_x = test_x.reshape(-1, 3072)
    test_y = np.asarray(dataset["test_batch"][b"labels"])
    test_set = (test_x, test_y)

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))
    return train_set, test_set


def cifar100(data_dir, one_hot=False):
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    checksum = "eb9058c3a382ffc7106e4002c42a8d85"

    save_path = os.path.join(data_dir, url.split("/")[-1])
    print("Preparing CIFAR-100 dataset...")
    try:
        download_url(url, save_path, checksum)
    except Exception as e:
        print("Error downloading dataset: %s" % str(e))
        sys.exit(1)

    # load the dataset
    dataset = {}
    with open(save_path, "rb") as f:
        tar = tarfile.open(fileobj=f)
        for item in tar:
            obj = tar.extractfile(item)
            if not obj or item.size < 100:
                continue
            cont = pickle.load(obj, encoding="bytes")
            dataset[item.name.split("/")[-1]] = cont

    train_x = dataset["train"][b"data"]
    train_x = train_x / 255.0
    train_y = np.asarray(dataset["train"][b"fine_labels"])
    train_set = (train_x, train_y)

    test_x = dataset["test"][b"data"]
    test_x = test_x / 255.0
    test_y = np.asarray(dataset["test"][b"fine_labels"])
    test_set = (test_x, test_y)

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))
    return train_set, test_set
