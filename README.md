<p align="center">
<img src="https://i.loli.net/2019/09/24/VatMxSiXLdg8yrD.png"/>
</p>

## About

tinynn is a lightweight deep learning framework designed with simplicity in mind. It is currently written in Python 3 using Numpy.

The `mini` branch contains the minimal components to run a neural network. The `master` branch has the latest stable code with more components and features. See [Components](#components) below.

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

- layers: Dense, Convolution2D, MaxPool2D, Dropout
- activation: ReLU, LeakyReLU, Sigmoid, Tanh
- losses: SoftmaxCrossEntropy, SigmoidCrossEntropy, MAE, MSE, Huber
- optimizer: SGD, Adam, Momentum, RMSProp

## Contribute

Please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for Python coding style.

In addition, please sort the module import order alphabetically in each file. To do this, one can use tools like [isort](https://github.com/timothycrosley/isort) (be sure to use `--force-single-line-imports` option to enforce the coding style).

## License

MIT
