
## tinynn


### Basic

tinynn 是一个基于 python 和 Numpy 实现的轻量级、易于扩展的深度学习框架。相关介绍看[这篇文章](https://borgwang.github.io/dl/2019/08/18/tinynn.html)

这个代码库有主要有两个分支，`mini` 分支实现了最基本最核心的框架组件，`master` 分支在 `mini` 分支的基础上进行扩展，支持更多的功能

### Getting Started

#### Install

```bash
git clone https://github.com/borgwang/tinynn.git
cd tinynn
pip install -r requirment.txt
```

#### Examples

```bash
cd tinynn
# MNIST classification
python examples/mnist/run.py  
# a toy regression task
python examples/nn_paint/run.py  
 # reinforcement learning demo (gym environment required)
python examples/rl/run.py 
```

#### License

MIT

