import os
import pickle

import numpy as np

from .buffers import Buffer
from .envs import Env
from .wrappers import LQRMultiBodyEnvWrapper, Wrapper


class Agent(object):
    def __init__(self, env: Env | Wrapper) -> None:
        self.env = env
        self.is_evaluate: bool = False

    def reset(self, **kwargs):
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train(self, buffer: Buffer, **kwargs):
        pass

    def save(self, savedir: str):
        pass

    def evaluate_mode(self, is_evaluate: bool):
        self.is_evaluate = is_evaluate


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
        """課題1: Ricacci方程式の解Pを求める
        以下にRicacci方程式を解くプログラムを実装してください．(READMEの解説を参照してください)
        """
        self.K = (-np.linalg.inv(self.R) @ B.T @ P).real

    def act(self, obs: np.ndarray) -> np.ndarray:
        return (self.K @ obs).astype(self.env.action_space.dtype)

    def save(self, savedir: str):
        _savedir = os.path.join(savedir, "agent")
        os.makedirs(_savedir, exist_ok=True)
        lqr_data = {"Q": self.Q, "R": self.R, "K": self.K}
        with open(os.path.join(_savedir, "lqr_data.pickle"), "wb") as f:
            pickle.dump(lqr_data, f)
