# -*- coding: utf-8 -*-

import os
from pt_implement.dqn.dqn import DeepQNetwork
from env.maze_env import Maze


def train():
    dqn = DeepQNetwork(env, max_timesteps, env.n_actions, env.n_features, learning_rate, gamma,
                       replace_target_iter, memory_size, batch_size=32, learning_starts=200,
                       train_freq=5, exploration_fraction=0.1, exploration_final_eps=0.02,
                       ckpt_dir=ckpt_dir)

    dqn.learn()
    print('game over')
    env.destroy()

    dqn.plot_cost()

if __name__ == "__main__":
    max_timesteps = 20000
    learning_rate = 0.01
    gamma = 0.9
    replace_target_iter = 200
    memory_size = 5000
    ckpt_dir = os.path.join(os.path.curdir, "ckpt", "model")

    env = Maze()

    train()