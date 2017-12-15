# -*- coding:utf-8 -*-

import gym
import tensorflow as tf
from env.arm_env import ArmEnv
import os
from tf_implement.a3c.a3c import ACNet


def enjoy():
    with tf.Session() as sess:
        opt_a = tf.train.RMSPropOptimizer(lr_a, name='RMSPropA')
        opt_c = tf.train.RMSPropOptimizer(lr_c, name='RMSPropC')
        global_ac = ACNet(sess, global_net_scope,
                          ns=ns, na=na, a_bound=a_bound,
                          opt_a=opt_a, opt_c=opt_c)  # we only need its params

        AC = ACNet(sess, "W_0", ns, na, a_bound, opt_a, opt_c, entropy_beta, global_ac)

        saver = tf.train.Saver()

        saver.restore(sess, os.path.join(ckpt_dir, "ckpt1", "model"))

        var = 0  # control exploration
        while True:
            s = env.reset()
            ep_reward = 0
            for j in range(max_ep_step):
                env.render()
                # Add exploration noise
                a = AC.choose_action(s)
                # a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
                s_, r, done, info = env.step(a)

                s = s_
                ep_reward += r
                if done or j == max_ep_step - 1:
                    print(' Reward: %i' % int(ep_reward))
                    break

if __name__ == "__main__":
    MODE = ['easy', 'hard']
    env = ArmEnv(mode=MODE[1])
    ns = env.state_dim
    na = env.action_dim
    a_bound = env.action_bound
    global_net_scope = 'global_net'
    lr_a = 0.0001
    lr_c = 0.001
    gamma = 0.9
    max_ep_step = 300
    max_global_ep = 2000
    update_global_iter = 10
    entropy_beta = 0.01
    lr_a = 0.0001
    lr_c = 0.001
    ckpt_dir = os.path.curdir
    enjoy()