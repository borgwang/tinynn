"""
Example code for visualizing feature maps and dense layers of
convolutional network using Activation Maximization (AM).
The subject Network is a pre-trained LeNet-5 on MNIST.

Reference
[1] Visualizing Higher-Layer Features of a Deep Network (Dumitru
    Erhan, et al.)
[2] Deep Inside Convolutional Networks: Visualising Image Classi-
    fication Models and Saliency Maps (Karen Simonyan, et al.)
"""

import runtime_path  # isort:skip

import argparse
import gzip
# import os
import pickle
# import sys
# import time
# 
# import numpy as np
# 
# from core.evaluator import AccEvaluator
# from core.layers import Conv2D
# from core.layers import Dense
# from core.layers import Flatten
# from core.layers import MaxPool2D
# from core.layers import ReLU
# from core.losses import SoftmaxCrossEntropyLoss
# from core.model import Model
# from core.nn import Net
# from core.optimizer import Adam
# from utils.data_iterator import BatchIterator
# from utils.downloader import download_url
# from utils.seeder import random_seed

def main(args):
    if args.seed >= 0:
        random_seed(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./examples/mnist/data", type=str)
    parser.add_argument("--model_path", default="./lenet-relu.pkl", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    main(args)
