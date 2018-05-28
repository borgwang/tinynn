# Author: borgwang <borgwang@126.com>
# Date: 2018-05-22
#
# Filename: run_dqn.py
# Description: DQN example code for demonstration


import os
import sys
sys.path.append(os.getcwd())
import gym
import numpy as np

from agent import DQN


def main(args):
    np.random.seed(args.seed)
    env = gym.make('CartPole-v0')
    agent = DQN(env, args)
    agent.construct_model()

    # load pretrained models or init new a model.
    saver = tf.train.Saver(max_to_keep=1)
    if args.model_path is not None:
        saver.restore(agent.sess, args.model_path)
        ep_base = int(args.model_path.split('_')[-1])
        best_mean_rewards = float(args.model_path.split('/')[-1].split('_')[0])
    else:
        agent.sess.run(tf.global_variables_initializer())
        ep_base = 0
        best_mean_rewards = None

    rewards_history, steps_history = [], []
    train_steps = 0
    # Training
    for ep in range(args.max_ep):
        state = env.reset()
        ep_rewards = 0
        for step in range(env.spec.timestep_limit):
            # pick action
            action = agent.sample_action(state, policy='egreedy')
            # Execution action.
            next_state, reward, done, debug = env.step(action)
            train_steps += 1
            ep_rewards += reward
            # modified reward to speed up learning
            reward = 0.1 if not done else -1
            # Learn and Update net parameters
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            if done:
                break
        steps_history.append(train_steps)
        if not rewards_history:
            rewards_history.append(ep_rewards)
        else:
            rewards_history.append(
                rewards_history[-1] * 0.9 + ep_rewards * 0.1)
        # Decay epsilon
        if agent.epsilon > args.final_epsilon:
            agent.epsilon -= (args.init_epsilon - args.final_epsilon) / args.max_ep

        # Evaluate during training
        if ep % args.log_every == args.log_every-1:
            total_reward = 0
            for i in range(args.test_ep):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    action = agent.sample_action(state, policy='greedy')
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            current_mean_rewards = total_reward / args.test_ep
            print('Episode: %d Average Reward: %.2f' %
                  (ep + 1, current_mean_rewards))
            # save model if current model outpeform the old one
            if best_mean_rewards is None or (current_mean_rewards >= best_mean_rewards):
                best_mean_rewards = current_mean_rewards
                if not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_name = args.save_path + str(round(best_mean_rewards, 2)) \
                    + '_' + str(ep_base + ep + 1)
                saver.save(agent.sess, save_name)
                print('Model saved %s' % save_name)

    # plot training rewards
    plt.plot(steps_history, rewards_history)
    plt.xlabel('steps')
    plt.ylabel('running avg rewards')
    plt.show()


def args_parse():
    # TODO: finish DQN example
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default=None,
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--save_path', default='examples/data/rl_models/',
        help='Path to save a model during training.')
    parser.add_argument(
        '--double_q', default=True, help='enable or disable double dqn')
    parser.add_argument(
        '--log_every', default=500, help='Log and save model every x episodes')
    parser.add_argument(
        '--seed', default=31, help='random seed')

    parser.add_argument(
        '--max_ep', type=int, default=2000, help='Number of training episodes')
    parser.add_argument(
        '--test_ep', type=int, default=50, help='Number of test episodes')
    parser.add_argument(
        '--init_epsilon', type=float, default=0.75, help='initial epsilon')
    parser.add_argument(
        '--final_epsilon', type=float, default=0.2, help='final epsilon')
    parser.add_argument(
        '--buffer_size', type=int, default=50000, help='Size of memory buffer')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Size of training batch')
    parser.add_argument(
        '--gamma', type=float, default=0.99, help='Discounted factor')
    parser.add_argument(
        '--target_network_update', type=int, default=1000,
        help='update frequency of target network.')
    return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
