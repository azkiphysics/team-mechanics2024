import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# Matplotlibで綺麗な論文用のグラフを作る
# https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
plt.rcParams["font.family"] = "Times New Roman"  # font familyの設定
plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます。
plt.rcParams["xtick.labelsize"] = 15  # 軸だけ変更されます。
plt.rcParams["ytick.labelsize"] = 15  # 軸だけ変更されます
plt.rcParams["xtick.direction"] = "in"  # x axis in
plt.rcParams["ytick.direction"] = "in"  # y axis in
plt.rcParams["axes.linewidth"] = 1.0  # axis line width
plt.rcParams["axes.grid"] = True  # make grid


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

    def integral(self, t: float, x: np.ndarray, u: np.ndarray) -> Tuple[float, np.ndarray]:
        """運動方程式の積分"""
        if self.integral_method == "euler_method":
            return self.euler_method(t, x, u)
        elif self.integral_method == "runge_kutta_method":
            return self.runge_kutta_method(t, x, u)
        else:
            assert False

    def euler_method(self, t: float, x: np.ndarray, u: np.ndarray) -> Tuple[float, np.ndarray]:
        """オイラー法"""
        dx_dt = self.motion_equation(x, u)
        next_t = t + self.dt
        next_x = x + self.dt * dx_dt
        return next_t, next_x

    def runge_kutta_method(self, t: float, x: np.ndarray, u: np.ndarray) -> Tuple[float, np.ndarray]:
        """ルンゲクッタ法"""
        k1 = self.motion_equation(t, x, u)
        k2 = self.motion_equation(t, x + self.dt / 2 * k1, u)
        k3 = self.motion_equation(t, x + self.dt / 2 * k2, u)
        k4 = self.motion_equation(t, x + self.dt * k3, u)
        next_t = t + self.dt
        next_x = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_t, next_x

    def motion_equation(self, t: float, x: np.ndarray, u: np.ndarray):
        """運動方程式"""
        raise NotImplementedError()

    def reset(
        self,
        initial_t: float,
        initial_x: np.ndarray,
        integral_method: str = "runge_kutta_method",
    ) -> Dict[str, bool | float | np.ndarray]:
        """シミュレーションの初期化"""
        self.integral_method = integral_method
        self.t = initial_t
        self.x = initial_x.copy()
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info

    def step(self, u: np.ndarray) -> Dict[str, bool | float | np.ndarray]:
        """シミュレーションの実行 (1ステップ)"""
        self.t, self.x = self.integral(self.t, self.x, u)
        done = self.t >= self.t_max
        info = {"t": self.t, "x": self.x.copy(), "done": done}
        return info


class MultiBodyEnv(Env):
    def compute_mass_matrix(self, t: float, x: np.ndarray) -> np.ndarray:
        """質量行列の計算"""
        raise NotImplementedError()

    def compute_external_force(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """外力の計算"""
        raise NotImplementedError()

    def compute_C(self, t: float, x: np.ndarray) -> np.ndarray:
        """拘束条件の計算"""
        raise NotImplementedError()

    def compute_Ct(self, t: float, x: np.ndarray) -> np.ndarray:
        """拘束条件の時間微分の計算"""
        raise NotImplementedError()

    def compute_Cq(self, t: float, x: np.ndarray) -> np.ndarray:
        """拘束条件のヤコビ行列の計算"""
        raise NotImplementedError()

    def compute_Ctt(self, t: float, x: np.ndarray) -> np.ndarray:
        """拘束条件の時間の2階微分の計算"""
        raise NotImplementedError()

    def compute_Cqt(self, t: float, x: np.ndarray) -> np.ndarray:
        """拘束条件のヤコビ行列の時間微分の計算"""
        raise NotImplementedError()

    def compute_Cqdqq(self, t: float, x: np.ndarray) -> np.ndarray:
        """(Cq * dq_dt)qの計算"""
        raise NotImplementedError()

    def get_independent_indices(self) -> List[int]:
        """独立変数のインデックスの取得 (位置座標qのインデックスを利用している)"""
        raise NotImplementedError()

    def newton_raphson_method(self, t: float, x: np.ndarray) -> np.ndarray:
        """ニュートンラフソン法"""
        pos_indices = list(range(0, len(x), 2))
        vel_indices = list(range(1, len(x), 2))
        independent_indices = self.get_independent_indices()
        Id = np.identity(4, dtype=np.float64)[independent_indices]
        while True:
            prev_x = x.copy()
            C = self.compute_C(t, x)
            Cq = self.compute_Cq(t, x)
            F = np.concatenate([Cq, Id], axis=0)
            f = np.concatenate([-C, np.zeros(2, dtype=np.float64)])
            dq = np.linalg.inv(F) @ f
            x[pos_indices] += dq
            if np.linalg.norm(x - prev_x) < 1e-11:
                break
        Ct = self.compute_Ct(t, x)
        Cq = self.compute_Cq(t, x)
        F = np.concatenate([Cq, Id], axis=0)
        f = np.concatenate([-Ct, x[vel_indices][independent_indices]])
        x[vel_indices] = np.linalg.inv(F) @ f
        return x

    def reset(
        self,
        initial_t: float,
        initial_x: np.ndarray,
        integral_method: str = "runge_kutta_method",
    ) -> Dict[str, bool | float | np.ndarray]:
        """シミュレーションの初期化"""
        info = super().reset(initial_t, initial_x, integral_method)
        self.x = self.newton_raphson_method(self.t, self.x)
        C = self.compute_C(self.t, self.x)
        info |= {"x": self.x.copy(), "C": C.copy()}
        return info

    def step(self, u) -> Dict[str, bool | float | np.ndarray]:
        """シミュレーションの実行 (1ステップ)"""
        info = super().step(u)
        self.x = self.newton_raphson_method(self.t, self.x)
        C = self.compute_C(self.t, self.x)
        info |= {"x": self.x.copy(), "C": C.copy()}
        return info


class CartPoleEnv(MultiBodyEnv):
    def __init__(
        self,
        t_max: float,
        dt: float = 0.001,
        m_cart: float = 1.0,
        m_ball: float = 1.0,
        l_pole: float = 1.0,
    ):
        super().__init__(t_max, dt)
        self.m_cart = m_cart  # カートの質量
        self.m_ball = m_ball  # ポールの上部に取り付けられたボールの質量
        self.l_pole = l_pole  # ポールの長さ
        self.g = 9.8  # 重力加速度

    def compute_mass_matrix(self, t: float, x: np.ndarray) -> np.ndarray:
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

    def compute_external_force(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        Q = np.zeros(6, dtype=np.float64)
        Q[0] = u[0]
        Q[2] = -self.m_ball * self.g
        dq = x[[1, 3, 5, 7]]
        Ctt = self.compute_Ctt(t, x)
        Cqdqq = self.compute_Cqdqq(t, x)
        Cqt = self.compute_Cqt(t, x)
        Q[4:] = -Ctt - Cqdqq @ dq - 2.0 * Cqt @ dq
        return Q

    def compute_C(self, t: float, x: np.ndarray) -> np.ndarray:
        C = np.zeros(2, dtype=np.float64)
        C[0] = x[2] - x[0] - self.l_pole * np.cos(x[6])
        C[1] = x[4] - self.l_pole * np.sin(x[6])
        return C

    def compute_Ct(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    def compute_Cq(self, t: float, x: np.ndarray) -> np.ndarray:
        Cq = np.zeros((2, 4), dtype=np.float64)
        Cq[0, 0] = -1.0
        Cq[0, 1] = 1.0
        Cq[0, 3] = self.l_pole * np.sin(x[6])
        Cq[1, 2] = 1.0
        Cq[1, 3] = -self.l_pole * np.cos(x[6])
        return Cq

    def compute_Ctt(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    def compute_Cqt(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.zeros((2, 4), dtype=np.float64)

    def compute_Cqdqq(self, t: float, x: np.ndarray) -> np.ndarray:
        Cqdqq = np.zeros((2, 4), dtype=np.float64)
        Cqdqq[0, 3] = x[7] * self.l_pole * np.cos(x[6])
        Cqdqq[1, 3] = x[7] * self.l_pole * np.sin(x[6])
        return Cqdqq

    def get_independent_indices(self) -> List[int]:
        return [0, 3]

    def motion_equation(self, t: float, x: np.ndarray, u: np.ndarray):
        dx_dt = np.zeros(8, dtype=np.float64)
        M = self.compute_mass_matrix(t, x)
        Q = self.compute_external_force(t, x, u)
        q_lam = np.linalg.inv(M) @ Q
        dx_dt[[0, 2, 4, 6]] = x[[1, 3, 5, 7]].copy()
        dx_dt[[1, 3, 5, 7]] = q_lam[:4].copy()
        return dx_dt


class FigureMaker(object):
    def __init__(self) -> None:
        self.fig: Figure = None
        self.ax: Axes = None

    def reset(self):
        """図作成ツールの初期化"""
        if self.ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.cla()

    def make(self, data: Dict[str, Dict[str, np.ndarray | List[Dict[str, np.ndarray]]]]):
        """図の作成"""
        x = data.get("x")
        y = data.get("y")
        x_label = x.get("label", "")
        x_value = x.get("value")
        y_label = y.get("label", "")
        y_value = y.get("value")

        for y_value_idx in y_value:
            self.ax.plot(x_value, y_value_idx.get("value"), label=y_value_idx.get("label", ""))
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.legend(loc="upper right")

    def save(self, savedir: str, savefile: str = "trajectory.png"):
        """図の保存"""
        if self.fig is not None:
            savepath = os.path.join(savedir, savefile)
            self.fig.savefig(savepath, dpi=300)

    def close(self):
        """matplotlibを閉じる"""
        plt.close()


if __name__ == "__main__":
    # シミュレーションの設定
    t_max = 10.0
    dt = 1e-3
    initial_t = 0.0
    initial_x = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.pi / 2 - 0.05, 0.0], dtype=np.float64)

    # シミュレーション環境の作成
    env = CartPoleEnv(t_max, dt=dt)
    buffer = Buffer()

    # シミュレーションの実行
    info = env.reset(initial_t, initial_x)
    done = info.pop("done")
    buffer.reset()
    buffer.push(info)
    while not done:
        u = np.zeros(1, dtype=np.float64)
        info = env.step(u)
        done = info.pop("done")
        buffer.push(info)

    # データの保存
    savedir = "result"
    savefile = "trajectory.pickle"
    buffer.save(savedir, savefile)

    # 図作成ツールの作成
    figure_maker = FigureMaker()
    data = buffer.get()

    # 状態変数の時間発展の描画
    t = np.array(data.get("t"), dtype=np.float64)
    independent_x = np.array(data.get("x"), dtype=np.float64)[:, [0, 1, 6, 7]]
    figure_data = {
        "x": {"label": "Time $t$ s", "value": t},
        "y": {
            "label": "State",
            "value": [
                {"label": "$x$", "value": independent_x[:, 0]},
                {"label": "$v_x$", "value": independent_x[:, 1]},
                {"label": "$\\theta$", "value": independent_x[:, 2]},
                {"label": "$\\omega$", "value": independent_x[:, 3]},
            ],
        },
    }
    figure_maker.reset()
    figure_maker.make(figure_data)
    figure_maker.save(savedir, savefile="traj_independent_x.png")

    # 拘束条件の時間発展の描画
    C = np.array(data.get("C"), dtype=np.float64)
    figure_data = {
        "x": {"label": "Time $t$ s", "value": t},
        "y": {
            "label": "Constraint $C$",
            "value": [
                {"label": "$C_1$", "value": C[:, 0]},
                {"label": "$C_2$", "value": C[:, 1]},
            ],
        },
    }
    figure_maker.reset()
    figure_maker.make(figure_data)
    figure_maker.save(savedir, savefile="traj_constraint.png")
