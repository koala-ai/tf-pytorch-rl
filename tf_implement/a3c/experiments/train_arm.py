# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import threading
from tf_implement.a3c.a3c import ACNet
from tf_implement.a3c.worker import Worker
from env.arm_env import ArmEnv


def train():
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        with tf.device("/cpu:0"):
            opt_a = tf.train.RMSPropOptimizer(lr_a, name='RMSPropA')
            opt_c = tf.train.RMSPropOptimizer(lr_c, name='RMSPropC')
            global_ac = ACNet(sess, global_net_scope,
                              ns=ns, na=na, a_bound=a_bound,
                              opt_a=opt_a, opt_c=opt_c)  # we only need its params
            workers = []
            # Create worker
            for i in range(n_nets):
                i_name = 'W_%i' % i  # worker name
                workers.append(Worker(sess, coord=coord, env=env, name=i_name, ns=ns, na=na, a_bound=a_bound,
                                      gamma=gamma, max_ep_step=max_ep_step, max_global_ep=max_global_ep,
                                      update_global_iter=update_global_iter, entropy_beta=entropy_beta, opt_a=opt_a,
                                      opt_c=opt_c, ckpt_dir=ckpt_dir, global_ac=global_ac))

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            worker_threads = []
            for worker in workers:
                job = lambda: worker.work()
                t = threading.Thread(target=job)
                t.start()
                worker_threads.append(t)
            coord.join(worker_threads)

        print("saved model")
        saver.save(sess, os.path.join(ckpt_dir, "ckpt1", "model"))

if __name__ == "__main__":
    n_nets = 3
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
    train()