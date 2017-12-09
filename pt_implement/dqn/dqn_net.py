# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch


class DQN_MLP(torch.nn.Module):
    def __init__(self, in_features, num_actions):
        super(DQN_MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)