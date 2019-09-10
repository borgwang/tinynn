"""Simple utilities to download and save a file with progress bar."""

import os
from urllib.error import URLError
from urllib.request import urlretrieve


def show_progress(blk_num, blk_sz, tot_sz):
    percentage = 100. * blk_num * blk_sz / tot_sz
    print('Progress: %.1f %%' % percentage, end='\r', flush=True)


def download_url(url, file_path):
    # create directory if needed
    Dir = os.path.dirname(file_path)
    if not os.path.exists(Dir):
        os.makedirs(Dir)
    # download
    try:
        if os.path.exists(file_path):
            print("{} already exists.".format(file_path))
        else:
            print("Downloading {} to {}".format(url, file_path))
            try:
                urlretrieve(url, file_path, show_progress)
            except URLError:
                raise RuntimeError("Error downloading resource!")
            finally:
                print()
    except KeyboardInterrupt:
        print("Interrupted")
