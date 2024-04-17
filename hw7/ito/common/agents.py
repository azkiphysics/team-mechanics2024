import numpy as np

from utils import Box, Discrete


class Agent(object):
    def __init__(self, observation_space: Box | Discrete, action_space: Box | Discrete, *args, **kwargs) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *args, **kwargs):
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        pass


class ZeroAgent(Agent):
    def __init__(self, observation_space: Box | Discrete, action_space: Box, *args, **kwargs) -> None:
        assert isinstance(action_space, Box)
        super().__init__(observation_space, action_space, *args, **kwargs)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = np.zeros(self.action_space.shape[0], dtype=self.action_space.dtype)
        return action
