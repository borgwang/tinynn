
## tinynn 

[![Build Status](https://travis-ci.com/borgwang/tinynn.svg?branch=master)](https://travis-ci.com/borgwang/tinynn)


tinynn is a lightweight deep learning framework written in Python3 (with NumPy).

<p align="center">
  <img src="http://ww4.sinaimg.cn/large/006tNc79gy1g63tkgdh1pj30to0fwjsk.jpg" width="80%" alt="tinynn-architecture" referrerPolicy="no-referrer"/>
</p>

There are two branches (`master` and `mini`) in thie repo. The `mini` branch contains the minimal components to run a neural network. The `master` branch holds the latest stable code with more components and features. See [Components](#components) down below.


## Getting Started

### Install

```bash
git clone https://github.com/borgwang/tinynn.git
cd tinynn
pip install -r requirements.txt
```

### Examples

```bash
cd tinynn
# MNIST classification
python examples/mnist/run.py  

# a toy regression task
python examples/nn_paint/run.py  

# reinforcement learning demo (gym environment required)
python examples/rl/run.py
```

### Components

- layers: Dense, Conv2D, ConvTranspose2D, MaxPool2D, Dropout, BatchNormalization
- activation: ReLU, LeakyReLU, Sigmoid, Tanh, Softplus
- losses: SoftmaxCrossEntropy, SigmoidCrossEntropy, MAE, MSE, Huber
- optimizer: RAdam, Adam, SGD, Momentum, RMSProp, Adagrad, Adadelta

## Contribute 

Please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for Python coding style.

In addition, please sort the module import order alphabetically in each file. To do this, one can use tools like [isort](https://github.com/timothycrosley/isort) (be sure to use `--force-single-line-imports` option to enforce the coding style).

## License

MIT
