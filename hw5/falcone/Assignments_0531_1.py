import os
import pickle

import numpy as np


class Buffer(object):
    def __init__(self) -> None:
        self.buffer = None

    def reset(self):
        self.buffer = {}

    def push(self, data):
        for key, value in data.items():
            if key not in self.buffer:
                self.buffer[key] = []
            self.buffer[key].append(value)

    def get(self):
        return self.buffer

    def save(self, savedir, savefile):
        buffer = self.get()
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, savefile)
        with open(path, "wb") as f:
            pickle.dump(buffer, f)


class Env(object):
    def __init__(self, t_max, dt=1e-3):
        self.t_max = t_max
        self.dt = dt

        self.t = None
        self.x = None

    def integral(self, x):
        k1 = self.motion_equation(x)
        k2 = self.motion_equation(x + self.dt / 2 * k1)
        k3 = self.motion_equation(x + self.dt / 2 * k2)
        k4 = self.motion_equation(x + self.dt * k3)
        next_x = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_x

    def motion_equation(self, x):
        return np.zeros_like(x, dtype=np.float64)

    def reset(self, initial_t, initial_x):
        self.t = initial_t
        self.x = initial_x.copy()
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info

    def step(self):
        self.t += self.dt
        self.x = self.integral(self.x)
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info


class LorenzEnv(Env):
    def __init__(self, t_max, dt=1e-3, rho=28.0, sigma=10.0, beta=8.0/3.0):
        super().__init__(t_max, dt=dt)
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

    def motion_equation(self, x):
        dx_dt = np.zeros_like(x, dtype=np.float64)
        dx_dt[0] = self.sigma * (x[1] - x[0])
        dx_dt[1] = x[0] * (self.rho - x[2]) -x[1]
        dx_dt[2] = x[0] * x[1] -self.beta * x[2]
        return dx_dt


if __name__ == "__main__":
    t_max = 50.0
    dt = 1e-3
    initial_t = 0.0
    initial_x = np.ones(3, dtype=np.float64)

    env = LorenzEnv(t_max, dt=dt)
    buffer = Buffer()

    info = env.reset(initial_t, initial_x)
    done = info.pop("done")
    buffer.reset()
    buffer.push(info)
    while not done:
        info = env.step()
        done = info.pop("done")
        buffer.push(info)

    savedir = "result"
    savefile = "trajectory.pickle"
    buffer.save(savedir, savefile)
