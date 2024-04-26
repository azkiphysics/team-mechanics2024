import os
import pickle

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

    def save(self, savedir: str):
        pass


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
        """課題1: Ricacci代数方程式の解を求める
        有本-ポッターの方法を利用して，Ricacci代数方程式を解き，
        Pを求めるコードを実装してください．
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
