# -*- coding: utf-8 -*-
import tensorflow as tf


class DQN_MLP(object):
    def __init__(self, state, n_actions, w_initializer, b_initializer,
                 scope_name, layer_name, dueling=False, trainable=True):
        self.state = state
        self.n_actions = n_actions
        self.w_initializer = w_initializer
        self.b_initializer = b_initializer
        self.scope_name = scope_name
        self.layer_name = layer_name
        self.dueling = dueling
        self.trainable = trainable

    def build(self):
        with tf.variable_scope(self.scope_name):
            e1 = tf.layers.dense(self.state,
                                 20,
                                 tf.nn.relu,
                                 kernel_initializer=self.w_initializer,
                                 bias_initializer=self.b_initializer,
                                 name=self.layer_name + str(1),
                                 trainable=self.trainable)
            if self.dueling:
                with tf.variable_scope('Value'):
                    v = tf.layers.dense(e1,
                                        1,
                                        kernel_initializer=self.w_initializer,
                                        bias_initializer=self.b_initializer,
                                        name=self.layer_name + str(2),
                                        trainable=self.trainable)
                with tf.variable_scope('Advantage'):
                    a = tf.layers.dense(e1,
                                        self.n_actions,
                                        kernel_initializer=self.w_initializer,
                                        bias_initializer=self.b_initializer,
                                        name=self.layer_name + str(3),
                                        trainable=self.trainable)
                q = v + (a - tf.reduce_mean(a, axis=1, keep_dims=True))
            else:
                q = tf.layers.dense(e1,
                                    self.n_actions,
                                    kernel_initializer=self.w_initializer,
                                    bias_initializer=self.b_initializer,
                                    name=self.layer_name + str(2),
                                    trainable=self.trainable)
        return q