# -*- coding: utf-8 -*-

from tf_implement.actor_critic.model import AC
import os
import gym


def enjoy():
    ac = AC(n_f=n_f, n_a=n_a, lr_a=lr_a, lr_c=lr_c)
    ac.saver.restore(ac.sess, os.path.join(ckpt_dir, "ckpt", "model_400"))

    while True:
        s = env.reset()
        track_r = []
        while True:
            env.render()
            a = ac.actor.choose_action(s)
            s_, r, done, info = env.step(a)

            track_r.append(r)

            if done:
                ep_rs_sum = sum(track_r)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                print("reward:", int(running_reward))
                break

if __name__ == "__main__":

    lr_a = 0.001
    lr_c = 0.01
    ckpt_dir = os.path.curdir

    env = gym.make('CartPole-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped

    n_f = env.observation_space.shape[0]
    n_a = env.action_space.n

    enjoy()


