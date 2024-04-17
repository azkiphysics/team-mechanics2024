from typing import Dict, Tuple
import numpy as np

from .envs import Env, MultiBodyEnv
from .utils import Box


class Wrapper(object):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def x_space(self) -> Box:
        return self.env.x_space

    @property
    def u_space(self) -> Box:
        return self.env.u_space

    @property
    def state_space(self) -> Box:
        return self.env.state_space

    @property
    def observation_space(self) -> Box:
        return self.env.observation_space

    @property
    def action_space(self) -> Box:
        return self.env.action_space

    def get_state(self, t: float, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        return self.env.get_state(t, x, u=u)

    def get_observation(self, t: float, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        return self.get_state(t, x, u=u)

    def get_control_input(self, action: np.ndarray) -> np.ndarray:
        return self.env.get_control_input(action)

    def get_reward(self, t: float, x: np.ndarray, u: np.ndarray) -> float:
        return self.env.get_reward(t, x, u)

    def get_terminated(self, t: float, x: np.ndarray, u: np.ndarray) -> bool:
        return self.env.get_terminated(t, x, u)

    def get_truncated(self, t: float, x: np.ndarray, u: np.ndarray) -> bool:
        return self.env.get_truncated(t, x, u)

    def reset(
        self, initial_t: float, initial_x: np.ndarray, integral_method: str = "runge_kutta_method", **kwargs
    ) -> Tuple[np.ndarray, Dict[str, bool | float | np.ndarray]]:
        _, info = self.env.reset(initial_t, initial_x, integral_method=integral_method, **kwargs)
        t = info.get("t")
        x = info.get("x")
        obs = self.get_observation(t, x)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, bool | float | np.ndarray]]:
        _, _, _, _, info = self.env.step(action)
        t = info.get("t")
        x = info.get("x")
        u = info.get("u")
        next_obs = self.get_observation(t, x, u)
        reward = self.get_reward(t, x, u)
        terminated = self.get_terminated(t, x, u)
        truncated = self.get_truncated(t, x, u)
        return next_obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self.env.render()

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped


class MultiBodyEnvWrapper(Wrapper):
    def __init__(self, env: MultiBodyEnv) -> None:
        self.env = env

        self.initial_t: float = None
        self.target_x: np.ndarray = None

    def get_state(self, t: float, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        state_indices = self.env.unwrapped.get_state_indices()
        state = x[state_indices].astype(self.state_space.dtype)
        target_state = self.target_x[state_indices].astype(self.state_space.dtype)
        return state - target_state

    def reset(
        self,
        initial_t: float,
        initial_x: np.ndarray,
        target_x: np.ndarray,
        integral_method: str = "runge_kutta_method",
        **kwargs,
    ) -> Tuple[np.ndarray | Dict[str, bool | float | np.ndarray]]:
        self.initial_t = initial_t
        self.target_x = self.env.unwrapped.newton_raphson_method(initial_t, target_x)
        return super().reset(initial_t=initial_t, initial_x=initial_x, integral_method=integral_method, **kwargs)

    @property
    def unwrapped(self) -> MultiBodyEnv:
        return self.env.unwrapped


class LQRMultiBodyEnvWrapper(MultiBodyEnvWrapper):
    def __init__(self, env: MultiBodyEnv) -> None:
        super().__init__(env)

        self.Q: np.ndarray = None
        self.R: np.ndarray = None

    def compute_ddqi(self, t: float, x: np.ndarray, u: np.ndarray):
        pos_indices = self.env.unwrapped.get_coordinate_indices()
        independent_pos_indices = self.env.unwrapped.get_independent_coordinate_indices()
        dependent_pos_indices = self.env.unwrapped.get_dependent_coordinate_indices()

        # Biの計算
        Cq = self.env.unwrapped.compute_Cq(t, x)
        Cqi = Cq[:, independent_pos_indices]
        Cqd = Cq[:, dependent_pos_indices]
        Cdi = -np.linalg.inv(Cqd) @ Cqi
        Bi = np.zeros((len(pos_indices), len(independent_pos_indices)), dtype=np.float64)
        Bi[independent_pos_indices] = np.identity(len(independent_pos_indices))
        Bi[dependent_pos_indices] = Cdi

        # ddqの計算
        M = self.env.unwrapped.compute_mass_matrix(t, x)[: len(pos_indices), : len(pos_indices)]
        Q = self.env.unwrapped.compute_external_force(t, x, u)
        Qe = Q[: len(pos_indices)]
        Qd = Q[len(pos_indices) :]
        Cd = np.linalg.inv(Cqd) @ Qd
        gammai = np.zeros(len(pos_indices), dtype=np.float64)
        gammai[dependent_pos_indices] = Cd
        return np.linalg.inv(Bi.T @ M @ Bi) @ (Bi.T @ Qe - Bi.T @ M @ gammai)

    @property
    def A(self) -> np.ndarray:
        assert self.initial_t is not None and self.target_x is not None
        state_indices = self.env.unwrapped.get_state_indices()
        n_states = len(state_indices)
        A = np.zeros((n_states, n_states), dtype=np.float64)
        A[: n_states // 2, n_states // 2 :] = np.identity(n_states // 2)
        u = np.zeros(self.env.unwrapped.u_space.shape[0], dtype=self.env.unwrapped.u_space.dtype)
        h = 1e-7
        for idx in range(n_states):
            target_x_plus = self.target_x.copy()
            target_x_plus[state_indices[idx]] += h
            ddqi_plus = self.compute_ddqi(self.initial_t, target_x_plus, u)

            target_x_minus = self.target_x.copy()
            target_x_minus[state_indices[idx]] -= h
            ddqi_minus = self.compute_ddqi(self.initial_t, target_x_minus, u)

            A[n_states // 2 :, idx] = (ddqi_plus - ddqi_minus) / (2 * h)
        return A

    @property
    def B(self) -> np.ndarray:
        assert self.initial_t is not None and self.target_x is not None
        state_indices = self.env.unwrapped.get_state_indices()
        n_states = len(state_indices)
        n_us = self.env.unwrapped.u_space.shape[0]
        B = np.zeros((n_states, n_us), dtype=np.float64)
        u = np.zeros(n_us, dtype=self.env.unwrapped.u_space.dtype)
        h = 1e-7
        for idx in range(n_us):
            u_plus = u.copy()
            u_plus[idx] += h
            ddqi_plus = self.compute_ddqi(self.initial_t, self.target_x, u_plus)

            u_minus = u.copy()
            u_minus[idx] -= h
            ddqi_minus = self.compute_ddqi(self.initial_t, self.target_x, u_minus)

            B[n_states // 2 :, idx] = (ddqi_plus - ddqi_minus) / (2 * h)
        return B

    @property
    def C(self) -> np.ndarray:
        n_states = self.state_space.shape[0]
        n_obs = self.observation_space.shape[0]
        return np.eye(n_obs, n_states, dtype=np.float64)

    @property
    def D(self) -> np.ndarray:
        n_obs = self.observation_space.shape[0]
        n_us = self.env.unwrapped.u_space.shape[0]
        return np.zeros((n_obs, n_us), dtype=np.float64)

    def get_reward(self, t: float, x: np.ndarray, u: np.ndarray) -> float:
        s = self.get_state(t, x, u=u)
        cost = 0.5 * (s @ self.Q @ s + u @ self.R @ u) * self.env.unwrapped.dt
        return -cost

    def reset(
        self,
        initial_t: float,
        initial_x: np.ndarray,
        target_x: np.ndarray,
        Q: float | np.ndarray,
        R: float | np.ndarray,
        integral_method: str = "runge_kutta_method",
        **kwargs,
    ) -> Tuple[np.ndarray | Dict[str, bool | float | np.ndarray]]:
        n_obs = self.observation_space.shape[0]
        n_us = self.u_space.shape[0]
        self.Q = Q * np.identity(n_obs, dtype=np.float64)
        self.R = R * np.identity(n_us, dtype=np.float64)
        return super().reset(
            initial_t=initial_t, initial_x=initial_x, target_x=target_x, integral_method=integral_method, **kwargs
        )
