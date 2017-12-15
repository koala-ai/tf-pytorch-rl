# -*- coding: utf-8 -*-

from tf_implement.actor_critic.model import AC
import os
import gym


def train():
    ac = AC(n_f=n_f, n_a=n_a, lr_a=lr_a, lr_c=lr_c)
    ac.init_variables()

    for i_episode in range(max_eposide):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            env.render()
            a = ac.actor.choose_action(s)
            s_, r, done, info = env.step(a)

            if done:
                r = -20
            track_r.append(r)
            td_error = ac.critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            ac.actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            t += 1
            if done or t >= max_ep_steps:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                print("episode:", i_episode, "  reward:", int(running_reward))
                break
        if i_episode % 100 == 0:
            ac.saver.save(ac.sess, os.path.join(ckpt_dir, "ckpt", "model_" + str(i_episode)))

if __name__ == "__main__":
    max_eposide = 500
    max_ep_steps = 1000

    lr_a = 0.001
    lr_c = 0.01
    ckpt_dir = os.path.curdir

    env = gym.make('CartPole-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped

    n_f = env.observation_space.shape[0]
    n_a = env.action_space.n

    train()


