# -*- coding: utf-8 -*-

import os
import gym
import numpy as np
from tf_implement.dqn.dueling_dqn import DuelingDQN


def predict():
    dueling_DQN = DuelingDQN(env, max_timesteps, action_space, n_features, learning_rate, gamma,
                       replace_target_iter, memory_size, batch_size=32, learning_starts=200,
                       train_freq=5, exploration_fraction=0.1, exploration_final_eps=0.02, ckpt_dir=ckpt_dir)

    dueling_DQN.saver.restore(dueling_DQN.sess, ckpt_dir + "_9900")

    while True:
        observation = env.reset()
        while True:
            env.render()
            action = dueling_DQN.choose_action(observation, None, eps=0.0)
            f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)
            observation_, reward, done, info = env.step(np.array([f_action]))
            observation = observation_


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

    predict()