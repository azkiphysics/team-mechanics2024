from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import Box, Discrete

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class DQNPolicy(nn.Module):
    def __init__(self, observation_space: Box, action_space: Discrete) -> None:
        super(DQNPolicy, self).__init__()

        n_observations = observation_space.shape[0]
        n_actions = action_space.n
        self.h1 = nn.Linear(n_observations, 64)
        self.activation1 = nn.ReLU()
        self.h2 = nn.Linear(64, 64)
        self.activation2 = nn.ReLU()
        self.out = nn.Linear(64, n_actions)

        nn.init.orthogonal_(self.h1.weight, gain=1.0)
        nn.init.orthogonal_(self.h2.weight, gain=1.0)
        self.h1.bias.data.fill_(0.0)
        self.h2.bias.data.fill_(0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h1 = self.activation1(self.h1(obs))
        h2 = self.activation2(self.h2(h1))
        return self.out(h2)


class DDPGActor(nn.Module):
    def __init__(self, observation_space: Box, action_space: Box) -> None:
        super(DDPGActor, self).__init__()

        n_observations = observation_space.shape[0]
        n_actions = action_space.shape[0]
        self.h1 = nn.Linear(n_observations, 400)
        self.activation1 = nn.ReLU()
        self.h2 = nn.Linear(400, 300)
        self.activation2 = nn.ReLU()
        self.out = nn.Linear(300, n_actions)
        self.activation_out = nn.Tanh()

        nn.init.orthogonal_(self.h1.weight, gain=1.0)
        nn.init.orthogonal_(self.h2.weight, gain=1.0)
        self.h1.bias.data.fill_(0.0)
        self.h2.bias.data.fill_(0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h1 = self.activation1(self.h1(obs))
        h2 = self.activation2(self.h2(h1))
        return self.activation_out(self.out(h2))


class DDPGCritic(nn.Module):
    def __init__(self, observation_space: Box, action_space: Box) -> None:
        super(DDPGCritic, self).__init__()

        n_observations = observation_space.shape[0]
        n_actions = action_space.shape[0]
        self.h1 = nn.Linear(n_observations + n_actions, 400)
        self.activation1 = nn.ReLU()
        self.h2 = nn.Linear(400, 300)
        self.activation2 = nn.ReLU()
        self.out = nn.Linear(300, 1)

        nn.init.orthogonal_(self.h1.weight, gain=1.0)
        nn.init.orthogonal_(self.h2.weight, gain=1.0)
        self.h1.bias.data.fill_(0.0)
        self.h2.bias.data.fill_(0.0)

    def forward(self, obs_action: torch.Tensor) -> torch.Tensor:
        h1 = self.activation1(self.h1(obs_action))
        h2 = self.activation2(self.h2(h1))
        return self.out(h2)


class SACActor(nn.Module):
    def __init__(self, observation_space: Box, action_space: Box) -> None:
        super(SACActor, self).__init__()

        n_observations = observation_space.shape[0]
        n_actions = action_space.shape[0]
        self.h1 = nn.Linear(n_observations, 256)
        self.activation1 = nn.ReLU()
        self.h2 = nn.Linear(256, 256)
        self.activation2 = nn.ReLU()
        self.mu = nn.Linear(256, n_actions)
        self.log_std = nn.Linear(256, n_actions)
        self.epsilon = 1e-6

        nn.init.orthogonal_(self.h1.weight, gain=1.0)
        nn.init.orthogonal_(self.h2.weight, gain=1.0)
        self.h1.bias.data.fill_(0.0)
        self.h2.bias.data.fill_(0.0)

    def forward(self, obs: torch.Tensor) -> Normal:
        h1 = self.activation1(self.h1(obs))
        h2 = self.activation2(self.h2(h1))
        mu = self.mu(h2)
        log_std = torch.clamp(self.log_std(h2), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        distribution = Normal(mu, std)
        return distribution

    def get_probabilistic_action(self, obs: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(obs)
        action = torch.tanh(distribution.rsample())
        return action

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(obs)
        action = torch.tanh(distribution.mean)
        return action

    def log_prob_pi(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.forward(obs)
        gaussian_action = distribution.rsample()
        action = torch.tanh(gaussian_action)
        log_prob = torch.sum(distribution.log_prob(gaussian_action), dim=1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - action**2 + self.epsilon), dim=1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, observation_space: Box, action_space: Box) -> None:
        super(SACCritic, self).__init__()

        n_observations = observation_space.shape[0]
        n_actions = action_space.shape[0]
        self.h1 = nn.Linear(n_observations + n_actions, 256)
        self.activation1 = nn.ReLU()
        self.h2 = nn.Linear(256, 256)
        self.activation2 = nn.ReLU()
        self.out = nn.Linear(256, 1)

        nn.init.orthogonal_(self.h1.weight, gain=1.0)
        nn.init.orthogonal_(self.h2.weight, gain=1.0)
        self.h1.bias.data.fill_(0.0)
        self.h2.bias.data.fill_(0.0)

    def forward(self, obs_action: torch.Tensor) -> torch.Tensor:
        h1 = self.activation1(self.h1(obs_action))
        h2 = self.activation2(self.h2(h1))
        return self.out(h2)
