from typing import Dict

import numpy as np

from utils import Box, Discrete


class Agent(object):
    def __init__(self, observation_space: Box | Discrete, action_space: Box | Discrete) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        pass

    def train(self, data: Dict[str, np.ndarray]):
        pass


class ZeroAgent(Agent):
    def act(self, obs: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, Box):
            action = np.zeros(self.action_space.shape[0], dtype=np.float64)
        elif isinstance(self.action_space, Discrete):
            action = np.zeros(self.action_space.n, dtype=np.int64)
        else:
            assert False
        return action
