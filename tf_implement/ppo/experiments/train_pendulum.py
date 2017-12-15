# -*- coding: utf-8 -*-

import threading
import os
import queue
import tensorflow as tf
from tf_implement.ppo.ppo import PPO
from tf_implement.ppo.ppo import Worker


def train():
    update_event, rolling_event = threading.Event(), threading.Event()
    update_event.clear()  # not update now
    rolling_event.set()  # start to roll out

    coord = tf.train.Coordinator()

    queues = queue.Queue()  # workers putting data in this queue

    global_ppo = PPO(s_dim, c_lr, a_dim, a_lr, epsilon, coord, ep_max,
                 update_event, queues, rolling_event, update_step)

    global_ppo.init_variable()

    workers = [Worker(i, game, global_ppo, coord, ep_len, rolling_event,
                      min_batch_size, gamma, ep_max, queues, update_event)
               for i in range(n_worker)]

    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=global_ppo.update, ))
    threads[-1].start()
    coord.join(threads)

    global_ppo.saver.save(global_ppo.sess, os.path.join(ckpt_dir, "ckpt", "model"))


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