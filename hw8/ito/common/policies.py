import torch
import torch.nn as nn

from .utils import Box, Discrete


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
