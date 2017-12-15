# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_implement.actor_critic.actor import Actor
from tf_implement.actor_critic.critic import Critic


class AC(object):
    def __init__(self, n_f, n_a, lr_a, lr_c):
        self.sess = tf.Session()
        self.actor = Actor(self.sess, n_features=n_f, n_actions=n_a, lr=lr_a)
        self.critic = Critic(self.sess, n_features=n_f, lr=lr_c)
        self.saver = tf.train.Saver()

    def init_variables(self):
        self.sess.run(tf.global_variables_initializer())
