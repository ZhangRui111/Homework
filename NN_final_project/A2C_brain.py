import numpy as np
import tensorflow as tf


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], 'state')
        self.a = tf.placeholder(tf.int32, None, 'act')
        self.td_error = tf.placeholder(tf.float32, None, 'td_error')  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.contrib.layers.fully_connected(self.s, 32, activation_fn=tf.nn.relu)
            self.acts_prob = tf.contrib.layers.fully_connected(l1, n_actions, activation_fn=tf.nn.softmax)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], 'state')
        self.v_ = tf.placeholder(tf.float32, [1, 1], 'v_next')
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.contrib.layers.fully_connected(self.s, 32, activation_fn=tf.nn.relu)
            self.v = tf.contrib.layers.fully_connected(l1, 1, activation_fn=None)

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + 0.9 * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error
