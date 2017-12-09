# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_implement.dqn.dqn_net import DQN_MLP
from utils.replay_buffer import ReplayBuffer
from utils.schedule import LinearSchedule


# 深度Q网络
class DeepQNetwork(object):
    def __init__(
            self,
            env,
            max_timesteps,
            n_actions,
            n_features,
            learning_rate,
            gamma,
            replace_target_iter,
            memory_size,
            batch_size,
            learning_starts,
            train_freq,
            exploration_fraction,
            exploration_final_eps,
            ckpt_dir=None):
        self.env = env
        self.max_timesteps = max_timesteps
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.ckpt_dir = ckpt_dir

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.replay_buffer = ReplayBuffer(self.memory_size)
        # self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * self.max_timesteps),
                                          initial_p=1.0,
                                          final_p=self.exploration_final_eps)

        # 参数替换
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.cost_his = []

    def init_variables(self):
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 当前状态
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # 下一个状态
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # 反馈
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # 行为

        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        # 评价网络，mlp网络
        self.q_eval =\
            DQN_MLP(self.s, self.n_actions, w_initializer, b_initializer, 'eval_net', 'e').build()

        # 目标网络，mlp网络
        self.q_next =\
            DQN_MLP(self.s_, self.n_actions,  w_initializer, b_initializer, 'target_net', 't', trainable=False).build()

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation, t, eps=None):
        if eps is not None:
            eps_threshold = eps
        else:
            eps_threshold = self.exploration.value(t)
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if np.random.uniform() < eps_threshold:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        last_obs = self.env.reset()
        for step in range(self.max_timesteps):
            # fresh env
            self.env.render()

            # dqn choose action based on observation
            action = self.choose_action(last_obs, step)
            env_action = action

            # dqn take action and get next observation and reward
            obs, reward, done = self.env.step(env_action)

            self.replay_buffer.add(last_obs, action, reward, obs, float(done))
            last_obs = obs

            if (step > self.learning_starts) and (step % self.train_freq == 0):
                self.learn_batch()

            if done:
                last_obs = self.env.reset()

    def learn_batch(self):
        # sample batch memory from all memory
        obs_batch, act_batch, rew_batch, next_obs_batch, _ = self.replay_buffer.sample(self.batch_size)

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: obs_batch,
                self.a: act_batch,
                self.r: rew_batch,
                self.s_: next_obs_batch,
            })

        self.cost_his.append(cost)

        self.learn_step_counter += 1

        if self.learn_step_counter % 100 == 0:
            self.saver.save(self.sess, self.ckpt_dir + "_" + str(self.learn_step_counter))

        # 每隔replace_target_iter，替换目标网络的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('target params replaced\n')

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
