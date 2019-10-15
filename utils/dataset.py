"""Common datasets"""

import runtime_path  # isort:skip

import gzip
import os
import pickle
import sys

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
