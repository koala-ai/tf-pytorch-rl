# -*- coding: utf-8 -*-

import os
import gym
from tf_implement.ppo.ppo import PPO


def train():

    global_ppo = PPO(s_dim, c_lr, a_dim, a_lr, epsilon, None, ep_max,
                 None, None, None, update_step)

    global_ppo.saver.restore(global_ppo.sess, os.path.join(ckpt_dir, "ckpt", "model"))

    env = gym.make('Pendulum-v0')
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(global_ppo.choose_action(s))[0]


if __name__ == "__main__":
    ep_max = 1000
    ep_len = 200
    gamma = 0.9  # reward discount factor
    a_lr = 0.0001  # learning rate for actor
    c_lr = 0.0002  # learning rate for critic
    min_batch_size = 64  # minimum batch size for updating PPO
    update_step = 10  # loop update operation n-steps
    epsilon = 0.2  # for clipping surrogate objective
    game = 'Pendulum-v0'
    s_dim, a_dim = 3, 1  # state and action dimension
    n_worker = 4
    ckpt_dir = os.path.curdir
    train()