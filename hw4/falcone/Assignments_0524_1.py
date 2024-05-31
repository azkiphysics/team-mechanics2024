import csv
import pickle


def add_data(result, info):
    for key, value in info.items():
        if key in result:
            result[key].append(value)


class Env(object):
    def __init__(self, t_max, dt=1e-3):
        self.t_max = t_max
        self.dt = dt

        self.t = None
        self.x = None
        self.y = None
        self.z = None

    def integral(self, x, y, z):
        k1_x, k1_y, k1_z = self.motion_equation(x, y, z)
        k2_x, k2_y, k2_z = self.motion_equation(x + self.dt / 2 * k1_x, y + self.dt / 2 * k1_y, z + self.dt / 2 * k1_z)
        k3_x, k3_y, k3_z = self.motion_equation(x + self.dt / 2 * k2_x, y + self.dt / 2 * k2_y, z + self.dt / 2 * k2_z)
        k4_x, k4_y, k4_z = self.motion_equation(x + self.dt * k3_x, y + self.dt * k3_y, z + self.dt * k3_z)
        next_x = x + self.dt / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        next_y = y + self.dt / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        next_z = x + self.dt / 6 * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
        return next_x, next_y, next_z

    def motion_equation(self, x, y, z):
        return 0.0, 0.0, 0.0

    def reset(self, initial_t, initial_x, initial_y, initial_z):
        self.t = initial_t
        self.x, self.y, self.z = initial_x, initial_y, initial_z
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x, "y": self.y, "z": self.z, "done": done}
        return info

    def step(self):
        self.t += self.dt
        self.x, self.y, self.z = self.integral(self.x, self.y, self.z)
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x, "y": self.y, "z": self.z, "done": done}
        return info


class LorenzEnv(Env):
    def __init__(self, t_max, dt=1e-3, rho=28.0, sigma=10.0, beta=8.0/3.0):
        super().__init__(t_max, dt=dt)
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

    def motion_equation(self, x, y, z):
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return dx_dt, dy_dt, dz_dt


if __name__ == "__main__":
    t_max = 0.1
    dt = 1e-3
    initial_t = 0.0
    initial_x = 1.0
    initial_y = 1.0
    initial_z = 1.0

    result = {"t": [], "x": [], "y": [], "z": []}

    env = LorenzEnv(t_max, dt=dt)

    info = env.reset(initial_t, initial_x, initial_y, initial_z)
    done = info.pop("done")
    add_data(result, info) # データの保存
    while not done:
        info = env.step()
        done = info.pop("done")
        add_data(result, info) # データの保存

    # csvファイルの保存
    path = "hw1_result.csv"
    keys = list(result.keys())
    data = [[result[key][i] for key in keys] for i in range(len(result[keys[0]]))]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([keys] + data)
    
    # pickleファイルの保存
    path = "hw1_result.pickle"
    with open(path, "wb") as f:
        pickle.dump(result, f)