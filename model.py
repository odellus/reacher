# -*- coding: utf-8

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_cfg, device

cfg = load_cfg()

FC1_SIZE_ACTOR = cfg["Model"]["fc1_size_actor"]
FC2_SIZE_ACTOR = cfg["Model"]["fc2_size_actor"]

FCS1_SIZE_CRITIC = cfg["Model"]["fcs1_size_critic"]
FC2_SIZE_CRITIC = cfg["Model"]["fc2_size_critic"]
WEIGHT_INIT_LIM = cfg["Model"]["weight_init_lim"]

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1./ np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
        """Actor Policy model."""

        def __init__(
            self,
            state_size,
            action_size,
            seed,
            fc1_units=FC1_SIZE_ACTOR,
            fc2_units=FC2_SIZE_ACTOR
            ):
            """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each state
                seed (int): Random seed
                fc1_units (int): Number of nodes in first hidden layer
                fc2_units (int): Number of nodes in second hidden layer
            """
            super(Actor, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
            self.reset_parameters()

        def reset_parameters(self):
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-WEIGHT_INIT_LIM, WEIGHT_INIT_LIM)

        def forward(self, state):
            """Build an actor policy that maps states to actions."""
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic value model."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        fcs1_units=FCS1_SIZE_CRITIC,
        fc2_units=FC2_SIZE_CRITIC
        ):
        """Initialize parameters and build model.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1) # Mapping onto a single scalar value.
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-WEIGHT_INIT_LIM, WEIGHT_INIT_LIM)

    def forward(self, state, action):
        """Build a critic value network that maps state-action pairs to Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
