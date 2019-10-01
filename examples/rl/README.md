## Reinforcement learning example with tinynn

Implement a [DQN](https://arxiv.org/pdf/1312.5602.pdf) agent to solve game of [CartPole](https://gym.openai.com/envs/CartPole-v0/).

Here we use a 2-layers fully-connected network to fit the Q function since this task is relatively simple.

### Run

```bash
# in the project directory
pip install -r requirements.txt
pip install -r examples/rl/requirements.txt

# run
python examples/rl/run.py
```
