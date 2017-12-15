# -*- coding: utf-8 -*-

import os
import gym
from tf_implement.dqn.dueling_dqn import DuelingDQN


def train():
    dueling_DQN = DuelingDQN(env, max_timesteps, action_space, n_features, learning_rate, gamma,
                       replace_target_iter, memory_size, batch_size=32, learning_starts=200,
                       train_freq=5, exploration_fraction=0.1, exploration_final_eps=0.02, ckpt_dir=ckpt_dir)

    dueling_DQN.init_variables()

    dueling_DQN.learn()


if __name__ == "__main__":
    max_timesteps = 50000
    action_space = 25
    n_features = 3
    learning_rate = 0.01
    gamma = 0.9
    replace_target_iter = 200
    memory_size = 10000
    ckpt_dir = os.path.join(os.path.curdir, "ckpt3", "model")
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    train()