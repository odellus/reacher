# -*- coding: utf-8

import numpy as np
# import copy
from model import Actor, Critic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from model import Actor, Critic
from noise import OUNoise
from replay_buffer import ReplayBuffer
from utils import load_cfg, device

cfg = load_cfg()

# Define global configuration variables
BUFFER_SIZE = cfg["Agent"]["Buffer_size"]
BATCH_SIZE = cfg["Agent"]["Batch_size"]
GAMMA = cfg["Agent"]["Gamma"]
TAU = cfg["Agent"]["Tau"]
LR_ACTOR = cfg["Agent"]["Lr_actor"]
LR_CRITIC = cfg["Agent"]["Lr_critic"]
WEIGHT_DECAY = cfg["Agent"]["Weight_decay"]
NOISE_DECAY = cfg["Agent"]["Noise_decay"]
UPDATE_EVERY = cfg["Agent"]["Update_every"]
LEARNING_STEPS = cfg["Agent"]["Learning_steps"]
NORM_REWARDS = cfg["Agent"]["Norm_rewards"]


class Agent:
    """Initeracts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        # Actor Networks both Local and Target.
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Networks both Local and Target.
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_modulation = 1
        self.noise_decay = NOISE_DECAY

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Count number of steps
        self.n_steps = 0
        self.update_every = UPDATE_EVERY

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer
        to learn."""
        self.memory.add(state, action, reward, next_state, done)

        # Learn if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.n_steps % self.update_every == 0:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        self.noise_modulation *= self.noise_decay
        self.n_steps += 1

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise_modulation * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.n_steps = 0
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value paramters given batch of experience tuples.
        Q_targets = r + gamma * cirtic_target(next_state, actor_state(next)state)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        # Get predicted next-state actions and Q-values from target models.
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad() # Clear gradient
        critic_loss.backward()            # Backpropagation
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()      # Update parameters

        # Update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad() # Clear gradient
        actor_loss.backward()            # Backpropagation
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()      # Update parameters

        # Now we update the target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)



    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        theta_target = tau * theta_local + (1 - tau) * theta_target

        Params
        ======
            local_model: PyTorch model (weight source)
            target_model: PyTorch model (weight destination)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
