# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from pt_implement.dqn.dqn_net import DQN_MLP
from utils.replay_buffer import ReplayBuffer
from utils.schedule import LinearSchedule


class DeepQNetwork:
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

        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * self.max_timesteps),
                                          initial_p=1.0,
                                          final_p=self.exploration_final_eps)
        # total learning step
        self.learn_step_counter = 0

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.cost_his = []

    def _build_net(self):

        self.Q = DQN_MLP(self.n_features, self.n_actions).type(torch.FloatTensor)
        self.target_Q = DQN_MLP(self.n_features, self.n_actions).type(torch.FloatTensor)
        self.optimizer = optim.RMSprop(params=self.Q.parameters(), lr=self.lr)
        self.loss_func = torch.nn.MSELoss()

    def choose_action(self, observation, step, eps=None):
        observation = autograd.Variable(torch.unsqueeze(torch.FloatTensor(observation), 0))
        if eps is not None:
            eps_threshold = eps
        else:
            eps_threshold = self.exploration.value(step)
        # eps_threshold = self.exploration.value(step)
        if np.random.random() > eps_threshold:
            actions_value = self.Q.forward(observation)
            action = torch.max(actions_value, 1)[1].data.numpy()
            return action[0]
        else:
            return np.random.randint(0, self.n_actions)

    def learn(self):
        last_obs = self.env.reset()
        for step in range(self.max_timesteps):
            self.env.render()
            action = self.choose_action(last_obs, step)
            # Advance one step
            obs, reward, done = self.env.step(action)

            self.replay_buffer.add(last_obs, action, reward, obs, done)
            # Resets the environment when reaching an episode boundary.
            last_obs = obs

            if step > self.learning_starts and step % self.train_freq == 0:
                # Periodically update the target network by Q network to target Q network
                if self.learn_step_counter % self.replace_target_iter == 0:
                    self.target_Q.load_state_dict(self.Q.state_dict())
                    print('\ntarget_params_replaced\n')

                obs_batch, act_batch, rew_batch, next_obs_batch, _ = self.replay_buffer.sample(self.batch_size)
                # Convert numpy nd_array to torch variables for calculation
                obs_batch = autograd.Variable(torch.from_numpy(obs_batch).type(torch.FloatTensor))
                act_batch = autograd.Variable(torch.from_numpy(act_batch).type(torch.LongTensor))
                rew_batch = autograd.Variable(torch.from_numpy(rew_batch).type(torch.FloatTensor))
                next_obs_batch = autograd.Variable(torch.from_numpy(next_obs_batch).type(torch.FloatTensor))

                # q_eval w.r.t the action in experience
                q_eval = self.Q(obs_batch).gather(1, act_batch.unsqueeze(1))  # shape (batch, 1)
                q_next = self.target_Q(next_obs_batch).detach()  # detach from graph, don't backpropagate
                q_target = rew_batch + self.gamma * q_next.max(1)[0]  # shape (batch, 1)
                loss = self.loss_func(q_eval, q_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.learn_step_counter += 1
                self.cost_his.append(loss.data[0])

                if self.learn_step_counter % 100 == 0:
                    torch.save(self.Q.state_dict(), self.ckpt_dir + "_" + str(self.learn_step_counter))

            if done:
                self.env.reset()

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
