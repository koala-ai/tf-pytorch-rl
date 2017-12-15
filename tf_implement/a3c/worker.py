# -*- coding: utf-8 -*-

import numpy as np
import platform
from tf_implement.a3c.a3c import ACNet

GLOBAL_RUNNING_R = []
GLOBAL_EP = 0


class Worker(object):
    def __init__(self, sess, coord, env, name, ns, na, a_bound, gamma,
                 max_ep_step, max_global_ep, update_global_iter,
                 entropy_beta, opt_a, opt_c, ckpt_dir, global_ac=None):
        self.sess = sess
        self.coord = coord
        self.gamma = gamma
        self.max_ep_step = max_ep_step
        self.max_global_ep = max_global_ep
        self.update_global_iter = update_global_iter
        self.ckpt_dir = ckpt_dir
        self.env = env
        self.name = name
        self.AC = ACNet(sess, name, ns, na, a_bound,
                        opt_a, opt_c, entropy_beta, global_ac)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not self.coord.should_stop() and GLOBAL_EP < self.max_global_ep:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(self.max_ep_step):
                # OSX OpenCV allows rendering only in main thread
                if self.name == 'W_0' and platform.system() != "Darwin":
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == self.max_ep_step - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)    # normalize

                if total_step % self.update_global_iter == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + self.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break