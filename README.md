
## tinynn 

[![Build Status](https://travis-ci.com/borgwang/tinynn.svg?branch=master)](https://travis-ci.com/borgwang/tinynn)


tinynn is a lightweight deep learning framework written in Python3 (with NumPy).

<p align="center">
  <img src="http://ww4.sinaimg.cn/large/006tNc79gy1g63tkgdh1pj30to0fwjsk.jpg" width="80%" alt="tinynn-architecture" referrerPolicy="no-referrer"/>
</p>

## Getting Started

### Install

```bash
pip install tinynn
```

### Examples

```bash
git clone https://github.com/borgwang/tinynn.git
cd tinynn/examples

# MNIST classification
python mnist/run.py  

# a toy regression task
python nn_paint/run.py  

# reinforcement learning demo (gym environment required)
python rl/run.py
```

### Components

- layers: Dense, Conv2D, ConvTranspose2D, RNN, MaxPool2D, Dropout, BatchNormalization
- activation: ReLU, LeakyReLU, Sigmoid, Tanh, Softplus
- losses: SoftmaxCrossEntropy, SigmoidCrossEntropy, MAE, MSE, Huber
- optimizer: RAdam, Adam, SGD, Momentum, RMSProp, Adagrad, Adadelta

## Contribute 

Please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for Python coding style.

In addition, please sort the module import order alphabetically in each file. To do this, one can use tools like [isort](https://github.com/timothycrosley/isort) (be sure to use `--force-single-line-imports` option to enforce the coding style).

## License

MIT
