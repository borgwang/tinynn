# tinynn

[![CI](https://github.com/borgwang/tinynn/actions/workflows/workflow.yml/badge.svg)](https://github.com/borgwang/tinynn/actions/workflows/workflow.yml)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/borgwang/tinynn.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/borgwang/tinynn/context:python)
[![codecov](https://codecov.io/gh/borgwang/tinynn/branch/master/graph/badge.svg?token=XyLajHqYr9)](https://codecov.io/gh/borgwang/tinynn)

tinynn is a lightweight deep learning framework written in Python3 (for learning purposes).

## Getting Started

#### Install

```bash
pip3 install tinynn
```

#### Run examples

```bash
git clone https://github.com/borgwang/tinynn.git
cd tinynn/examples

# MNIST classification
python3 mnist/run.py

# a toy regression task
python3 nn_paint/run.py

# reinforcement learning demo (gym environment required)
python3 rl/run.py
```

#### Intuitive APIs

```python
# define a model
net = Net([Dense(50), ReLU(), Dense(100), ReLU(), Dense(10)])
model = Model(net=net, loss=MSE(), optimizer=Adam(lr))

# train
for batch in iterator(train_x, train_y):
    preds = model.forward(batch.inputs)
    loss, grads = model.backward(preds, batch.targets)
    model.apply_grads(grads)
```

## Contribute

Please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for Python coding style.

## License

MIT

