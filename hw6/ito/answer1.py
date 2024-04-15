import os
import pickle
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


class Buffer(object):
    def __init__(self) -> None:
        self.buffer: Dict[str, List[float | np.ndarray]] | None = None

    def reset(self):
        self.buffer = defaultdict(list)

    def push(self, data: Dict[str, List[float | np.ndarray]]):
        if self.buffer is None:
            self.reset()
        for key, value in data.items():
            self.buffer[key].append(value)

    def get(self) -> Dict[str, List[float | np.ndarray]] | None:
        return self.buffer

    def save(self, savedir: str, savefile: str):
        buffer = self.get()
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, savefile)
        with open(path, "wb") as f:
            pickle.dump(buffer, f)


class Env(object):
    def __init__(self, t_max: float, dt: float = 1e-3):
        self.t_max = t_max
        self.dt = dt

        self.t: float = None
        self.x: np.ndarray | None = None
        self.integral_method: str | None = None

    def integral(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self.integral_method == "euler_method":
            return self.euler_method(x, u)
        elif self.integral_method == "runge_kutta_method":
            return self.runge_kutta_method(x, u)
        else:
            assert False

    def euler_method(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        dx_dt = self.motion_equation(x, u)
        next_x = x + self.dt * dx_dt
        return next_x

    def runge_kutta_method(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        k1 = self.motion_equation(x, u)
        k2 = self.motion_equation(x + self.dt / 2 * k1, u)
        k3 = self.motion_equation(x + self.dt / 2 * k2, u)
        k4 = self.motion_equation(x + self.dt * k3, u)
        next_x = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_x

    def motion_equation(self, x: np.ndarray, u: np.ndarray):
        return np.zeros_like(x, dtype=np.float64)

    def reset(
        self,
        initial_t: float,
        initial_x: np.ndarray,
        integral_method: str = "runge_kutta_method",
    ) -> Dict[str, bool | float | np.ndarray]:
        self.integral_method = integral_method
        self.t = initial_t
        self.x = initial_x.copy()
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info

    def step(self, u: np.ndarray) -> Dict[str, bool | float | np.ndarray]:
        self.t += self.dt
        self.x = self.integral(self.x, u)
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info


class CartPoleEnv(Env):
    def __init__(
        self,
        t_max: float,
        dt: float = 0.001,
        m_cart: float = 1.0,
        m_ball: float = 1.0,
        l_pole: float = 1.0,
    ):
        super().__init__(t_max, dt)
        self.m_cart = m_cart
        self.m_ball = m_ball
        self.l_pole = l_pole
        self.g = 9.8

    def compute_mass_matrix(self, x: np.ndarray) -> np.ndarray:
        M = np.zeros((6, 6), dtype=np.float64)
        M[0, 0] = self.m_cart
        M[0, 4] = -1.0
        M[1, 1] = self.m_ball
        M[1, 4] = 1.0
        M[2, 2] = self.m_ball
        M[2, 5] = 1.0
        M[3, 4] = self.l_pole * np.sin(x[6])
        M[3, 5] = -self.l_pole * np.cos(x[6])
        M[4, 0] = -1.0
        M[4, 1] = 1.0
        M[4, 3] = self.l_pole * np.sin(x[6])
        M[5, 2] = 1.0
        M[5, 3] = -self.l_pole * np.cos(x[6])
        return M

    def compute_external_force(
        self, x: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        Q = np.zeros(6, dtype=np.float64)
        Q[0] = u[0]
        Q[2] = -self.m_ball * self.g
        Q[4] = -self.l_pole * np.cos(x[6]) * x[7] ** 2
        Q[5] = -self.l_pole * np.sin(x[6]) * x[7] ** 2
        return Q

    def compute_C(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                x[2] - x[0] - self.l_pole * np.cos(x[6]),
                x[4] - self.l_pole * np.sin(x[6]),
            ],
            dtype=np.float64,
        )

    def compute_Ct(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    def compute_Cq(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [-1.0, 1.0, 0.0, self.l_pole * np.sin(x[6])],
                [0.0, 0.0, 1.0, -self.l_pole * np.cos(x[6])],
            ],
            dtype=np.float64,
        )

    def newton_raphson_method(self, x: np.ndarray) -> np.ndarray:
        pos_indices = [0, 2, 4, 6]
        vel_indices = [1, 3, 5, 7]
        independent_indices = [0, 3]
        Id = np.identity(4, dtype=np.float64)[independent_indices]
        while True:
            prev_x = x.copy()
            C = self.compute_C(x)
            Cq = self.compute_Cq(x)
            F = np.concatenate([Cq, Id], axis=0)
            f = np.concatenate([-C, np.zeros(2, dtype=np.float64)])
            dq = np.linalg.inv(F) @ f
            x[pos_indices] += dq
            if np.linalg.norm(x - prev_x) < 1e-11:
                break
        Ct = self.compute_Ct(x)
        Cq = self.compute_Cq(x)
        F = np.concatenate([Cq, Id], axis=0)
        f = np.concatenate([-Ct, x[vel_indices][independent_indices]])
        x[vel_indices] = np.linalg.inv(F) @ f
        return x

    def motion_equation(self, x: np.ndarray, u: np.ndarray):
        dx_dt = np.zeros(8, dtype=np.float64)
        M = self.compute_mass_matrix(x)
        Q = self.compute_external_force(x, u)
        q_lam = np.linalg.inv(M) @ Q
        dx_dt[0] = x[1].copy()
        dx_dt[2] = x[3].copy()
        dx_dt[4] = x[5].copy()
        dx_dt[6] = x[7].copy()
        dx_dt[[1, 3, 5, 7]] = q_lam[:4].copy()
        return dx_dt

    def reset(
        self,
        initial_t: float,
        initial_x: np.ndarray,
        integral_method: str = "runge_kutta_method",
    ) -> Dict[str, bool | float | np.ndarray]:
        info = super().reset(initial_t, initial_x, integral_method)
        self.x = self.newton_raphson_method(self.x)
        info |= {"x": self.x.copy()}
        return info

    def step(self, u) -> Dict[str, bool | float | np.ndarray]:
        info = super().step(u)
        self.x = self.newton_raphson_method(self.x)
        info |= {"x": self.x.copy()}
        return info


class Figure(object):
    def __init__(self) -> None:
        pass

    def reset(
        self,
    ):
        self.fig, self.ax = plt.subplots()

    def make(
        self,
        data: Dict[str, List[float | np.ndarray]],
        indices: List[int],
        labels: List[str],
    ):
        t = data.get("t")
        x = data.get("x")
        if t is None or x is None:
            return
        x = np.vstack(x)
        for idx, label in zip(indices, labels):
            value = data["x"][:, idx]
            self.ax.plot(t, value, label=label)
        self.ax.legend(loc="upper right")

    def save(self, savedir: str):
        savefile = "trajectory.png"
        savepath = os.path.join(savedir, savefile)
        self.fig.savefig(savepath, dpi=300)

    def close(self):
        plt.close()


if __name__ == "__main__":
    t_max = 10.0
    dt = 1e-3
    initial_t = 0.0
    initial_x = np.array(
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.pi / 2 - 0.05, 0.0], dtype=np.float64
    )

    env = CartPoleEnv(t_max, dt=dt)
    buffer = Buffer()

    info = env.reset(initial_t, initial_x)
    done = info.pop("done")
    buffer.reset()
    buffer.push(info)
    while not done:
        u = np.zeros(1, dtype=np.float64)
        info = env.step(u)
        done = info.pop("done")
        buffer.push(info)

    savedir = "result"
    savefile = "trajectory.pickle"
    buffer.save(savedir, savefile)
