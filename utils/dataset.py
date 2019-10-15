"""Common datasets"""

import runtime_path  # isort:skip

import gzip
import os
import pickle
import sys
import tarfile

import numpy as np

from utils.downloader import download_url


def mnist(data_dir):
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
        return pickle.load(f, encoding="latin1")


def cifar10(data_dir):
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
    train_y = np.asarray(train_y)
    train_set = (train_x, train_y)

    test_batch_name = ["test_batch"]
    test_x = dataset["test_batch"][b"data"]
    test_y = np.asarray(dataset["test_batch"][b"labels"])
    test_set = (test_x, test_y)

    return (train_set, test_set)


def cifar100(data_dir):
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
    train_y = np.asarray(dataset["train"][b"fine_labels"])
    train_set = (train_x, train_y)

    test_x = dataset["test"][b"data"]
    test_y = np.asarray(dataset["test"][b"fine_labels"])
    test_set = (test_x, test_y)

    return (train_set, test_set)
