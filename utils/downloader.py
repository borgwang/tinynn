"""Simple utilities to download and save a file with progress bar."""

import hashlib
import os
from urllib.error import URLError
from urllib.request import urlretrieve


def show_progress(blk_num, blk_sz, tot_sz):
    percentage = 100. * blk_num * blk_sz / tot_sz
    print("Progress: %.1f %%" % percentage, end="\r", flush=True)


def md5_checksum(file_path):
    with open(file_path, "rb") as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    return checksum


def download_url(url, file_path, checksum):
    # create directory if needed
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if os.path.exists(file_path):
        # check md5
        if md5_checksum(file_path) == checksum:
            print("{} already exists.".format(file_path))
            return 
        else:
            print("Wrong checksum!")

    try:
        print("Downloading {} to {}".format(url, file_path))
        urlretrieve(url, file_path, show_progress)
    except URLError:
        raise RuntimeError("Error downloading resource!")
    except KeyboardInterrupt:
        print("Interrupted")
