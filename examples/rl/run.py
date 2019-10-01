"""Vanilla Deep Q Network (https://www.nature.com/articles/nature14236?wm=book_wap_0005) with tinynn."""

import runtime_path  # isort:skip

import argparse

import matplotlib.pyplot as plt

import gym
from examples.rl.agent import DQN
from utils.seeder import random_seed


def main(args):
    env = gym.make("CartPole-v0")

    if args.seed >= 0:
        random_seed(args.seed)
        env.seed(args.seed)


    agent = DQN(env, args)
    agent.construct_model()

    rewards_history, steps_history = [], []
    train_steps = 0
    # Training
    for ep in range(args.max_ep):
        state = env.reset()
        ep_rewards = 0
        for step in range(env.spec.timestep_limit):
            # pick action
            action = agent.sample_action(state, policy="egreedy")
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
                    if args.render:
                        env.render()
                    action = agent.sample_action(state, policy="greedy")
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            current_mean_rewards = total_reward / args.test_ep
            print("Episode: %d Average Reward: %.2f" %
                  (ep + 1, current_mean_rewards))

    # plot training rewards
    plt.plot(steps_history, rewards_history)
    plt.xlabel("steps")
    plt.ylabel("running avg rewards")
    plt.show()


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_every", default=100, help="Log and save model every x episodes")
    parser.add_argument(
        "--seed", default=-1, help="random seed")

    parser.add_argument(
        "--max_ep", type=int, default=1000, help="Number of training episodes")
    parser.add_argument(
        "--test_ep", type=int, default=5, help="Number of test episodes")
    parser.add_argument(
        "--init_epsilon", type=float, default=0.75, help="initial epsilon")
    parser.add_argument(
        "--final_epsilon", type=float, default=0.2, help="final epsilon")
    parser.add_argument(
        "--buffer_size", type=int, default=50000, help="Size of memory buffer")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Size of training batch")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discounted factor")
    parser.add_argument(
        "--render", type=bool, default=True, help="Render evaluation")
    parser.add_argument(
        "--target_network_update", type=int, default=1000,
        help="update frequency of target network.")
    return parser.parse_args()


if __name__ == "__main__":
    main(args_parse())
