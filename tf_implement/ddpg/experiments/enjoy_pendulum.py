# -*- coding: utf-8 -*-

import gym
import numpy as np
import os
from tf_implement.ddpg.ddpg import DDPG


def enjoy():
    ddpg = DDPG(env, max_epochs, train_steps, a_dim, s_dim, a_bound,
                memory_size, batch_size=batch_size,
                tau=tau, gamma=gamma, lr_a=lr_a, lr_c=lr_c, ckpt_dir=ckpt_dir)

    ddpg.saver.restore(ddpg.sess, ckpt_dir + "_198")

    while True:
        s = env.reset()
        while True:
            env.render()
            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, 0.1), -2, 2)
            s_, r, done, info = env.step(a)
            s = s_

if __name__ == "__main__":
    max_epochs = 200
    train_steps = 1000
    lr_a = 0.001
    lr_c = 0.002
    gamma = 0.9
    tau = 0.01
    memory_size = 10000
    batch_size = 32
    ckpt_dir = os.path.join(os.path.curdir, "ckpt", "model")

    env = gym.make("Pendulum-v0")
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    enjoy()