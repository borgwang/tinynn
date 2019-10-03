"""DQN agent class"""

import numpy as np
import random
from collections import deque

from core.net import Net
from core.layer import Dense, ReLU
from core.model import Model
from core.loss import MSE
from core.optimizer import RMSProp


class DQN(object):

    def __init__(self, env, args):
        self.args = args
        # Init replay buffer
        self.replay_buffer = deque(maxlen=args.buffer_size)

        # Init parameters
        self.global_step = 0
        self.epsilon = args.init_epsilon
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.gamma = args.gamma
        self.learning_rate = args.lr
        self.batch_size = args.batch_size

        self.target_network_update_interval = args.target_network_update

    def build_net(self):
        q_net = Net([
            Dense(100),
            ReLU(),
            Dense(self.action_dim)
        ])
        return q_net

    def construct_model(self):
        self.q_net = self.build_net()
        self.model = Model(net=self.q_net, loss=MSE(), optimizer=RMSProp(self.args.lr))
        # Target network
        self.target_q_net = self.build_net()

    def sample_action(self, state, policy):
        self.global_step += 1
        # Q value of all actions
        state = np.array([state])
        output_q = self.model.forward(state)[0]

        if policy == "egreedy":
            if random.random() <= self.epsilon:  # random action
                return random.randint(0, self.action_dim - 1)
            else:   # greedy action
                return np.argmax(output_q)
        elif policy == "greedy":
            return np.argmax(output_q)
        elif policy == "random":
            return random.randint(0, self.action_dim - 1)

    def learn(self, state, action, reward, next_state, done):
        onehot_action = np.zeros(self.action_dim)
        onehot_action[action] = 1

        # Store experience in deque
        self.replay_buffer.append(
            np.array([state, onehot_action, reward, next_state, done]))
        if len(self.replay_buffer) > self.batch_size:
            self.update_model()

    def update_model(self):
        if self.global_step % self.target_network_update_interval == 0:
            # Update target network. Assign params in q_net to target_q_net
            self.target_q_net.params = self.q_net.params

        # Sample experience
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        # Transpose minibatch
        s_batch, a_batch, r_batch, next_s_batch, done_batch = \
            np.array(minibatch).T.tolist()

        next_s_batch = np.array(next_s_batch)
        next_s_all_action_Q = self.target_q_net.forward(next_s_batch)
        next_s_Q_batch = np.max(next_s_all_action_Q, 1)

        # Calculate target_Q_batch
        target_Q_batch = []
        for i in range(self.batch_size):
            done_state = done_batch[i]
            if done_state:
                target_Q_batch.append(r_batch[i])
            else:
                target_Q_batch.append(
                    r_batch[i] + self.gamma * next_s_Q_batch[i])

        # Train the network
        preds = self.model.forward(np.asarray(s_batch))
        preds = np.multiply(preds, a_batch)

        targets = np.reshape(target_Q_batch, (-1, 1))
        targets = np.tile(targets, (1, 2))
        targets = np.multiply(targets, a_batch)
        loss, grads = self.model.backward(preds, targets)

        self.model.apply_grad(grads)
