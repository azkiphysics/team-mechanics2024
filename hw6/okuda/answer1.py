import argparse
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
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
plt.rcParams["axes.axisbelow"] = True  # グリッドを最背面に移動


class Buffer(object):
    """データ格納クラス"""

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
    """環境クラス (シミュレーション環境を作成するときは本クラスを親クラスとして実装する)"""

    def __init__(self, t_max: float, dt: float = 1e-3):
        self.t_max = t_max
        self.dt = dt

        self.t: float = None
        self.x: np.ndarray | None = None
        self.integral_method: str | None = None
        self.fig: Figure = None
        self.ax: Axes = None

    def integral(
        self, t: float, x: np.ndarray, u: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """運動方程式の積分"""
        if self.integral_method == "euler_method":
            return self.euler_method(t, x, u)
        elif self.integral_method == "runge_kutta_method":
            return self.runge_kutta_method(t, x, u)
        else:
            assert False

    def euler_method(
        self, t: float, x: np.ndarray, u: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """オイラー法"""
        next_t = t + self.dt
        """課題1: オイラー法
        以下にREADME.mdの解説に記載した更新式を利用してオイラー法を実装してください．
        積分後の状態変数xはnext_xとしてください．
        """
        next_x = x + self.motion_equation(t, x, u) * self.dt
        return next_t, next_x

    def runge_kutta_method(
        self, t: float, x: np.ndarray, u: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """課題1: ルンゲクッタ法
        以下にREADME.mdの解説に記載した更新式を利用してルンゲクッタ法を実装してください．
        積分後の状態変数xはnext_xとしてください．
        """
        k1 = self.motion_equation(t, x, u)
        k2 = self.motion_equation(t, x + self.dt / 2 * k1, u)
        k3 = self.motion_equation(t, x + self.dt / 2 * k2, u)
        k4 = self.motion_equation(t, x + self.dt * k3, u)
        next_x = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        next_t = t + self.dt
        return next_t, next_x

    def motion_equation(self, t: float, x: np.ndarray, u: np.ndarray):
        """運動方程式"""
        raise NotImplementedError()

    def compute_energy(self, t: float, x: np.ndarray) -> float:
        """全エネルギーの計算"""
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
        e = self.compute_energy(self.t, self.x)
        info = {"t": self.t, "x": self.x.copy(), "e": e, "done": done}
        return info

    def step(self, u: np.ndarray) -> Dict[str, bool | float | np.ndarray]:
        """シミュレーションの実行 (1ステップ)"""
        self.t, self.x = self.integral(self.t, self.x, u)
        done = self.t >= self.t_max
        e = self.compute_energy(self.t, self.x)
        info = {"t": self.t, "x": self.x.copy(), "e": e, "done": done}
        return info

    def render(self) -> np.ndarray:
        """図の描画"""
        raise NotImplementedError()


class MultiBodyEnv(Env):
    """マルチボディシステム用環境クラス"""

    def compute_mass_matrix(self, t: float, x: np.ndarray) -> np.ndarray:
        """質量行列の計算"""
        raise NotImplementedError()

    def compute_external_force(
        self, t: float, x: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
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

    def get_coordinate_indices(self) -> List[int]:
        """一般座標のインデックスの取得"""
        raise NotImplementedError()

    def get_independent_coordinate_indices(self) -> List[int]:
        """独立一般座標のインデックスの取得"""
        raise NotImplementedError()

    def get_velocity_indices(self) -> List[int]:
        """一般速度のインデックスの取得"""
        n_coordinates = len(self.get_coordinate_indices())
        return list(range(n_coordinates, 2 * n_coordinates))

    def get_independent_velocity_indices(self) -> List[int]:
        """独立一般速度のインデックスの取得"""
        n_coordinates = len(self.get_coordinate_indices())
        return [
            n_coordinates + idx for idx in self.get_independent_coordinate_indices()
        ]

    def get_state_indices(self) -> List[int]:
        """状態変数のインデックスの取得"""
        return (
            self.get_independent_coordinate_indices()
            + self.get_independent_velocity_indices()
        )

    def newton_raphson_method(self, t: float, x: np.ndarray) -> np.ndarray:
        """ニュートンラフソン法"""
        pos_indices = self.get_coordinate_indices()
        vel_indices = self.get_velocity_indices()
        independent_pos_indices = self.get_independent_coordinate_indices()
        independent_vel_indices = self.get_independent_velocity_indices()
        Id = np.identity(4, dtype=np.float64)[independent_pos_indices]
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
        f = np.concatenate([-Ct, x[independent_vel_indices]])
        x[vel_indices] = np.linalg.inv(F) @ f
        return x

    def reset(
        self,
        initial_t: float,
        initial_x: np.ndarray,
        integral_method: str = "runge_kutta_method",
    ) -> Dict[str, bool | float | np.ndarray]:
        """シミュレーションの初期化"""
        start = time.time()
        info = super().reset(initial_t, initial_x, integral_method)
        self.x = self.newton_raphson_method(self.t, self.x)
        e = self.compute_energy(self.t, self.x)
        C = self.compute_C(self.t, self.x)
        end = time.time()
        calc_speed = end - start
        info |= {"x": self.x.copy(), "e": e, "C": C.copy(), "calc_speed": calc_speed}
        return info

    def step(self, u) -> Dict[str, bool | float | np.ndarray]:
        """シミュレーションの実行 (1ステップ)"""
        start = time.time()
        info = super().step(u)
        self.x = self.newton_raphson_method(self.t, self.x)
        e = self.compute_energy(self.t, self.x)
        C = self.compute_C(self.t, self.x)
        end = time.time()
        calc_speed = end - start
        info |= {"x": self.x.copy(), "e": e, "C": C.copy(), "calc_speed": calc_speed}
        return info


class CartPoleEnv(MultiBodyEnv):
    """倒立振り子の環境クラス"""

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
        theta_pole = x[3]

        M = np.zeros((6, 6), dtype=np.float64)
        M[0, 0] = self.m_cart
        M[0, 4] = -1.0
        M[1, 1] = self.m_ball
        M[1, 4] = 1.0
        M[2, 2] = self.m_ball
        M[2, 5] = 1.0
        M[3, 4] = self.l_pole * np.sin(theta_pole)
        M[3, 5] = -self.l_pole * np.cos(theta_pole)
        M[4, 0] = -1.0
        M[4, 1] = 1.0
        M[4, 3] = self.l_pole * np.sin(theta_pole)
        M[5, 2] = 1.0
        M[5, 3] = -self.l_pole * np.cos(theta_pole)
        return M

    def compute_external_force(
        self, t: float, x: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        dq = x[4:]

        Q = np.zeros(6, dtype=np.float64)
        Q[0] = u[0]
        Q[2] = -self.m_ball * self.g
        Ctt = self.compute_Ctt(t, x)
        Cqdqq = self.compute_Cqdqq(t, x)
        Cqt = self.compute_Cqt(t, x)
        Q[4:] = -Ctt - Cqdqq @ dq - 2.0 * Cqt @ dq
        return Q

    def compute_C(self, t: float, x: np.ndarray) -> np.ndarray:
        x_cart, x_ball, y_ball, theta_pole = x[:4]

        C = np.zeros(2, dtype=np.float64)
        C[0] = x_ball - x_cart - self.l_pole * np.cos(theta_pole)
        C[1] = y_ball - self.l_pole * np.sin(theta_pole)
        return C

    def compute_Ct(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    def compute_Cq(self, t: float, x: np.ndarray) -> np.ndarray:
        theta_pole = x[3]

        Cq = np.zeros((2, 4), dtype=np.float64)
        Cq[0, 0] = -1.0
        Cq[0, 1] = 1.0
        Cq[0, 3] = self.l_pole * np.sin(theta_pole)
        Cq[1, 2] = 1.0
        Cq[1, 3] = -self.l_pole * np.cos(theta_pole)
        return Cq

    def compute_Ctt(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    def compute_Cqt(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.zeros((2, 4), dtype=np.float64)

    def compute_Cqdqq(self, t: float, x: np.ndarray) -> np.ndarray:
        theta_pole, theta_dot_pole = x[[3, 7]]

        Cqdqq = np.zeros((2, 4), dtype=np.float64)
        Cqdqq[0, 3] = theta_dot_pole * self.l_pole * np.cos(theta_pole)
        Cqdqq[1, 3] = theta_dot_pole * self.l_pole * np.sin(theta_pole)
        return Cqdqq

    def get_coordinate_indices(self) -> List[int]:
        return list(range(4))

    def get_independent_coordinate_indices(self) -> List[int]:
        return [0, 3]

    def motion_equation(self, t: float, x: np.ndarray, u: np.ndarray):
        independent_pos_indices = self.get_independent_coordinate_indices()
        independent_vel_indices = [4 + idx for idx in independent_pos_indices]

        dx_dt = np.zeros(8, dtype=np.float64)
        M = self.compute_mass_matrix(t, x)
        Q = self.compute_external_force(t, x, u)
        ddq_lam = np.linalg.inv(M) @ Q
        dx_dt[independent_pos_indices] = x[independent_vel_indices].copy()
        dx_dt[independent_vel_indices] = ddq_lam[independent_pos_indices].copy()
        return dx_dt

    def compute_energy(self, t: float, x: np.ndarray) -> float:
        y_ball, vx_cart, vx_ball, vy_ball = x[[2, 4, 5, 6]]
        return (
            0.5 * self.m_cart * vx_cart**2
            + 0.5 * self.m_ball * (vx_ball**2 + vy_ball**2)
            + self.m_ball * self.g * y_ball
        )

    def render(self) -> np.ndarray:
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        self.ax.cla()
        self.ax.set_xlim(-2.5 * self.l_pole, 2.5 * self.l_pole)
        self.ax.set_ylim(-1.5 * self.l_pole, 1.5 * self.l_pole)
        self.ax.set_axis_off()
        self.ax.set_aspect("equal")

        cart_center_x = self.x[0]
        ball_center = self.x[1:3]
        theta_pole = self.x[3]

        # 球体オブジェクト描画用角度配列
        angle_upper = np.linspace(0.0, np.pi, 100)
        angle_lower = np.linspace(0.0, -np.pi, 100)

        # 地平線の描画
        horizon_x = np.linspace(-10.0 * self.l_pole, 10.0 * self.l_pole, 100)
        horizon_y = -0.2 * self.l_pole * np.ones_like(horizon_x, dtype=np.float64)
        self.ax.plot(horizon_x, horizon_y, color="black")

        # カートの描画
        cart_x = np.linspace(
            -0.15 * self.l_pole + cart_center_x, 0.15 * self.l_pole + cart_center_x, 100
        )
        cart_y_upper = 0.1 * self.l_pole * np.ones_like(cart_x, dtype=np.float64)
        cart_y_lower = -0.1 * self.l_pole * np.ones_like(cart_x, dtype=np.float64)
        self.ax.fill_between(cart_x, cart_y_upper, cart_y_lower, color="blue", zorder=0)

        # 車輪の描画
        wheel_right_x = cart_center_x + self.l_pole * (0.1 + 0.05 * np.cos(angle_upper))
        wheel_left_x = cart_center_x + self.l_pole * (-0.1 + 0.05 * np.cos(angle_upper))
        wheel_upper_y = self.l_pole * (-0.15 + 0.05 * np.sin(angle_upper))
        wheel_lower_y = self.l_pole * (-0.15 + 0.05 * np.sin(angle_lower))
        self.ax.fill_between(
            wheel_right_x, wheel_upper_y, wheel_lower_y, color="black", zorder=1
        )
        self.ax.fill_between(
            wheel_left_x, wheel_upper_y, wheel_lower_y, color="black", zorder=1
        )

        # 球の描画
        ball_x = ball_center[0] + 0.1 * self.l_pole * np.cos(angle_upper)
        ball_upper_y = ball_center[1] + 0.1 * self.l_pole * np.sin(angle_upper)
        ball_lower_y = ball_center[1] + 0.1 * self.l_pole * np.sin(angle_lower)
        self.ax.fill_between(ball_x, ball_upper_y, ball_lower_y, color="red", zorder=2)

        # 棒の描画
        R = np.array(
            [
                [np.cos(theta_pole), -np.sin(theta_pole)],
                [np.sin(theta_pole), np.cos(theta_pole)],
            ],
            dtype=np.float64,
        )
        initial_pole_upper_x = np.array([0.0, 0.0], dtype=np.float64)
        initial_pole_lower_x = np.array([-self.l_pole, 0.0], dtype=np.float64)
        pole_upper_x, pole_upper_y = ball_center + R @ initial_pole_upper_x
        pole_lower_x, pole_lower_y = ball_center + R @ initial_pole_lower_x
        self.ax.plot(
            [pole_upper_x, pole_lower_x],
            [pole_upper_y, pole_lower_y],
            color="black",
            linewidth=2,
            zorder=1,
        )

        # 図の描画
        self.fig.canvas.draw()
        frame = np.array(self.fig.canvas.buffer_rgba())[:, :, :3]
        return frame


class FigureMaker(object):
    """図作成クラス"""

    def __init__(self) -> None:
        self.fig: Figure = None
        self.ax: Axes = None

    def reset(self):
        """図作成ツールの初期化"""
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
        else:
            self.ax.cla()

    def make(
        self, data: Dict[str, Dict[str, np.ndarray | List[Dict[str, np.ndarray]]]]
    ):
        """図の作成"""
        x = data.get("x")
        y = data.get("y")
        x_label = x.get("label", "")
        x_value = x.get("value")
        y_label = y.get("label", "")
        y_value = y.get("value")

        for y_value_idx in y_value:
            self.ax.plot(
                x_value, y_value_idx.get("value"), label=y_value_idx.get("label", "")
            )
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.legend(loc="upper right")

    def save(self, savedir: str, savefile: str = "trajectory.png"):
        """図の保存"""
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(savedir, savefile)
        self.fig.savefig(savepath, dpi=300)

    def close(self):
        """matplotlibを閉じる"""
        plt.close()


class MovieMaker(object):
    """動画作成クラス"""

    def __init__(self) -> None:
        self.frames: List[np.ndarray] = None

    def reset(self):
        self.frames = []

    def add(self, frame: np.ndarray):
        self.frames.append(frame)

    def make(self, savedir: str, t_max: float, savefile: str = "animation.mp4"):
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(savedir, savefile)
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        size = self.frames[0].shape[:2][::-1]
        video = cv2.VideoWriter(savepath, fourcc, int(len(self.frames) / t_max), size)
        for frame in self.frames:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)
        video.release()


if __name__ == "__main__":
    args = argparse.ArgumentParser("Cart pole problem")
    args.add_argument("integral_method", choices=["euler_method", "runge_kutta_method"])
    parser = args.parse_args()

    # シミュレーションの設定
    t_max = 10.0
    dt = 1e-3
    m_cart = 1.0
    m_ball = 1.0
    l_pole = 1.0
    initial_t = 0.0
    initial_x = np.array(
        [0.0, 0.0, 1.0, np.pi / 2 - 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    integral_method = parser.integral_method

    # シミュレーション環境の作成
    env = CartPoleEnv(t_max, dt=dt, m_cart=m_cart, m_ball=m_ball, l_pole=l_pole)
    buffer = Buffer()
    movie_maker = MovieMaker()

    # シミュレーションの実行
    info = env.reset(initial_t, initial_x, integral_method=integral_method)
    done = info.pop("done")
    buffer.reset()
    buffer.push(info)
    movie_maker.reset()
    movie_maker.add(env.render())
    k_steps = 0
    movie_freq = t_max // dt // 500
    while not done:
        k_steps += 1
        u = np.zeros(1, dtype=np.float64)
        info = env.step(u)
        done = info.pop("done")
        buffer.push(info)
        if k_steps % movie_freq == 0 or done:
            movie_maker.add(env.render())

    # データの保存
    savedir = os.path.join("result", integral_method)
    savefile = "trajectory.pickle"
    buffer.save(savedir, savefile)

    # 図作成ツールの作成
    figure_maker = FigureMaker()
    data = buffer.get()

    # 独立変数の時間発展の描画
    t = np.array(data.get("t"), dtype=np.float64)
    independent_state_indices = env.get_state_indices()
    independent_x = np.array(data.get("x"), dtype=np.float64)[
        :, independent_state_indices
    ]
    figure_data = {
        "x": {"label": "Time $t$ (s)", "value": t},
        "y": {
            "label": "Independent coordinates and velocities",
            "value": [
                {"label": "$x_{\\mathrm{cart}}$", "value": independent_x[:, 0]},
                {"label": "$\\theta_{\\mathrm{cart}}$", "value": independent_x[:, 1]},
                {"label": "$v_{\\mathrm{cart}}$", "value": independent_x[:, 2]},
                {"label": "$\\omega_{\\mathrm{cart}}$", "value": independent_x[:, 3]},
            ],
        },
    }
    figure_maker.reset()
    figure_maker.make(figure_data)
    figure_maker.save(savedir, savefile="traj_independent_x.png")

    # 拘束条件の時間発展の描画
    C = np.array(data.get("C"), dtype=np.float64)
    figure_data = {
        "x": {"label": "Time $t$ (s)", "value": t},
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

    # 全エネルギーの時間発展の描画
    e = np.array(data.get("e"), dtype=np.float64)
    figure_data = {
        "x": {"label": "Time $t$ (s)", "value": t},
        "y": {
            "label": "Total energy $E$ J",
            "value": [{"label": "$E$", "value": e}],
        },
    }
    figure_maker.reset()
    figure_maker.make(figure_data)
    figure_maker.save(savedir, savefile="traj_total_energy.png")

    # 計算速度の時間発展の描画
    calc_speed = np.array(data.get("calc_speed"), dtype=np.float64)
    figure_data = {
        "x": {"label": "Time $t$ (s)", "value": t},
        "y": {
            "label": "Calculation speed $t_{\\mathrm{calc}}$ (s)",
            "value": [{"label": "$t_{\\mathrm{calc}}$", "value": calc_speed}],
        },
    }
    figure_maker.reset()
    figure_maker.make(figure_data)
    figure_maker.save(savedir, savefile="traj_calculation_speed.png")
    figure_maker.close()

    # 動画の保存
    movie_maker.make(savedir, t[-1])
