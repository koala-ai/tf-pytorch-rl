# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from utils.replay_buffer import ReplayBuffer


class DDPG(object):
    def __init__(self,
                 env,
                 max_epochs,
                 train_steps,
                 a_dim,
                 s_dim,
                 a_bound,
                 memory_size,
                 batch_size,
                 tau,
                 gamma,
                 lr_c,
                 lr_a,
                 ckpt_dir=None):
        self.env = env
        self.max_epochs = max_epochs
        self.train_steps = train_steps
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.sess = tf.Session()
        self.learn_step_counter = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.replay_buffer = ReplayBuffer(self.memory_size)

        # 策略网络
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        # 值网络
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # 更新策略网络
        self.soft_replace = [[tf.assign(ta, (1 - tau) * ta + tau * ea), tf.assign(tc, (1 - tau) * tc + tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + gamma * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)

        self.saver = tf.train.Saver()

    def init_variables(self):
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        var = 3
        step = 0
        for epoch in range(self.max_epochs):
            last_obs = self.env.reset()

            for j in range(self.train_steps):
                # fresh env
                self.env.render()
                action = self.choose_action(last_obs)
                action = np.clip(np.random.normal(action, var), -2, 2)  # add randomness to action selection for exploration
                obs, reward, done, info = self.env.step(action)
                self.replay_buffer.add(last_obs, action, reward / 10, obs, float(done))
                last_obs = obs
                if step > 1000:
                    var *= 0.99999
                    self.learn_batch()
                step += 1

                if j == self.train_steps - 1:
                    print('Episode:', epoch, 'Explore: %.2f' % var, )
                    break

                if done:
                    self.env.reset()

            self.saver.save(self.sess, self.ckpt_dir + "_" + str(epoch))

    def learn_batch(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        obs_batch, act_batch, rew_batch, next_obs_batch, _ = self.replay_buffer.sample(self.batch_size)

        rew_batch = rew_batch[:, np.newaxis]

        self.sess.run(self.atrain, {self.S: obs_batch})
        self.sess.run(self.ctrain, {self.S: obs_batch, self.a: act_batch, self.R: rew_batch, self.S_: next_obs_batch})

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
