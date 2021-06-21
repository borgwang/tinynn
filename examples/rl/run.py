"""Implement vanilla Deep Q Network with tinynn."""

import argparse

import gym
import matplotlib.pyplot as plt
import tinynn as tn

from agent import DQN


def get_model(out_dim, lr):
    q_net = tn.net.Net([
        tn.layer.Dense(100),
        tn.layer.ReLU(),
        tn.layer.Dense(out_dim)
    ])
    return tn.model.Model(net=q_net, loss=tn.loss.MSE(), optimizer=tn.optimizer.RMSProp(lr))


def main(args):
    env = gym.make("CartPole-v0")

    if args.seed >= 0:
        tn.seeder.random_seed(args.seed)
        env.seed(args.seed)

    agent = DQN(env, args)
    model = get_model(out_dim=env.action_space.n, lr=args.lr)
    agent.set_model(model)

    rewards_history, steps_history = [], []
    train_steps = 0
    # Training
    for ep in range(args.max_ep):
        state = env.reset()
        ep_rewards = 0
        for _ in range(env.spec.timestep_limit):
            # sample action
            action = agent.sample_action(state, policy="egreedy")
            # apply action
            next_state, reward, done, _ = env.step(action)
            train_steps += 1
            ep_rewards += reward
            # modified reward to speed up learning
            reward = 0.1 if not done else -1
            # train
            agent.train(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

        steps_history.append(train_steps)
        if not rewards_history:
            rewards_history.append(ep_rewards)
        else:
            rewards_history.append(rewards_history[-1] * 0.9 + ep_rewards * 0.1)

        # Decay epsilon
        if agent.epsilon > args.final_epsilon:
            decay = (args.init_epsilon - args.final_epsilon) / args.max_ep
            agent.epsilon -= decay

        # Evaluate during training
        if ep % args.log_every == args.log_every-1:
            total_reward = 0
            for _ in range(args.test_ep):
                state = env.reset()
                for _ in range(env.spec.timestep_limit):
                    if args.render:
                        env.render()
                    action = agent.sample_action(state, policy="greedy")
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            current_mean_rewards = total_reward / args.test_ep
            print(f"Episode: {ep + 1} "
                  f"running reward: {current_mean_rewards:.2f}")

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
