import numpy as np
from .buffers import Buffer
from .envs import Env
from .wrappers import Wrapper, LQRMultiBodyEnvWrapper


class Agent(object):
    def __init__(self, env: Env | Wrapper) -> None:
        self.env = env

    def reset(self, **kwargs):
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train(self, buffer: Buffer, **kwargs):
        pass


class ZeroAgent(Agent):
    def __init__(self, env: Env | Wrapper, **kwargs) -> None:
        super().__init__(env)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.action_space.shape[0], dtype=self.env.action_space.dtype)
        return action


class LQRAgent(Agent):
    def __init__(self, env: LQRMultiBodyEnvWrapper) -> None:
        self.env = env

        self.Q = None
        self.R = None

    def reset(self, Q: float | np.ndarray, R: float | np.ndarray, **kwargs):
        A = self.env.A
        B = self.env.B
        self.Q = Q * np.identity(self.env.observation_space.shape[0], dtype=self.env.observation_space.dtype)
        self.R = R * np.identity(self.env.action_space.shape[0], dtype=self.env.action_space.dtype)
        AH = np.block([[A, -B @ np.linalg.inv(self.R) @ B.T], [-self.Q, -A.T]])
        """
        課題1: 有本-ポッターの方法を用いて，Ricacci代数方程式を解いて，
        Pを求めてください．(以下に実装を行なってください)
        """
        self.K = (-np.linalg.inv(self.R) @ B.T @ P).real

    def act(self, obs: np.ndarray) -> np.ndarray:
        return (self.K @ obs).astype(self.env.action_space.dtype)
