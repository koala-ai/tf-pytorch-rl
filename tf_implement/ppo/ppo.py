# -*- coding: utf-8 -*-

import tensorflow as tf
import gym
import numpy as np

GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
GLOBAL_RUNNING_R = []


class PPO(object):
    def __init__(self, s_dim, c_lr, a_dim, a_lr,
                 epsilon, coord, ep_max,
                 update_event, queue, rolling_event, update_step):
        self.sess = tf.Session()
        self.coord = coord
        self.a_dim = a_dim
        self.s_dim = s_dim
        # self.GLOBAL_EP = GLOBAL_EP
        self.ep_max = ep_max
        self.update_event = update_event
        self.queue = queue
        self.rolling_event = rolling_event
        self.update_step = update_step
        self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(c_lr).minimize(self.closs)

        # actor
        norm_dist, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(norm_dist.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = norm_dist.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(a_lr).minimize(self.aloss)
        self.saver = tf.train.Saver()

    def init_variable(self):
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EP
        while not self.coord.should_stop():
            if GLOBAL_EP < self.ep_max:
                self.update_event.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [self.queue.get() for _ in range(self.queue.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :self.s_dim], data[:, self.s_dim: self.s_dim + self.a_dim], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op,
                               {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.update_step)]

                [self.sess.run(self.ctrain_op,
                               {self.tfs: s, self.tfdc_r: r}) for _ in range(self.update_step)]
                self.update_event.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                self. rolling_event.set()         # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid, game, global_ppo, coord, ep_len, rolling_event,
                 min_batch_size, gamma, ep_max, queue, update_event):

        self.wid = wid
        self.coord = coord
        self.ep_len = ep_len
        self.rolling_event = rolling_event
        self.min_batch_size = min_batch_size
        self.gamma = gamma
        self.ep_max = ep_max
        self.queue = queue
        self.update_event = update_event
        self.env = gym.make(game).unwrapped
        self.ppo = global_ppo

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not self.coord.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(self.ep_len):
                if not self.rolling_event.is_set():                  # while global PPO is updating
                    self.rolling_event.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)                    # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                if t == self.ep_len - 1 or GLOBAL_UPDATE_COUNTER >= self.min_batch_size:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.queue.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= self.min_batch_size:
                        self.rolling_event.clear()       # stop collecting data
                        self.update_event.set()          # globalPPO update

                    if GLOBAL_EP >= self.ep_max:         # stop training
                        self.coord.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/self.ep_max * 100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)