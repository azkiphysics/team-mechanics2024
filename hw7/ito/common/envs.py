from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .utils import Box


class Env(object):
    """環境クラス (シミュレーション環境を作成するときは本クラスを親クラスとして実装する)"""

    def __init__(self, t_max: float, dt: float = 1e-3):
        self.t_max = t_max
        self.dt = dt

        self.initial_t: float = None
        self.t: float = None
        self.x: np.ndarray | None = None
        self.integral_method: str | None = None
        self.fig: Figure = None
        self.ax: Axes = None

    @property
    def x_space(self) -> Box:
        raise NotImplementedError()

    @property
    def u_space(self) -> Box:
        raise NotImplementedError()

    @property
    def state_space(self) -> Box:
        raise NotImplementedError()

    @property
    def observation_space(self) -> Box:
        raise NotImplementedError()

    @property
    def action_space(self) -> Box:
        raise NotImplementedError()

    def integral(self, t: float, x: np.ndarray, u: np.ndarray) -> tuple[float, np.ndarray]:
        """運動方程式の積分"""
        if self.integral_method == "euler_method":
            return self.euler_method(t, x, u)
        elif self.integral_method == "runge_kutta_method":
            return self.runge_kutta_method(t, x, u)
        else:
            assert False

    def euler_method(self, t: float, x: np.ndarray, u: np.ndarray) -> tuple[float, np.ndarray]:
        """オイラー法"""
        dx_dt = self.motion_equation(t, x, u)
        next_t = t + self.dt
        next_x = x + self.dt * dx_dt
        return next_t, next_x

    def runge_kutta_method(self, t: float, x: np.ndarray, u: np.ndarray) -> tuple[float, np.ndarray]:
        """ルンゲクッタ法"""
        k1 = self.motion_equation(t, x, u)
        k2 = self.motion_equation(t, x + self.dt / 2 * k1, u)
        k3 = self.motion_equation(t, x + self.dt / 2 * k2, u)
        k4 = self.motion_equation(t, x + self.dt * k3, u)
        next_t = t + self.dt
        next_x = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_t, next_x

    def get_state(self, x: np.ndarray) -> np.ndarray:
        """状態量の取得"""
        return x.astype(self.state_space.dtype)

    def get_observation(self, t: float, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """観測量の取得"""
        return self.get_state(x).astype(self.observation_space.dtype)

    def get_control_input(self, action: np.ndarray) -> np.ndarray:
        """制御入力の取得"""
        assert isinstance(action, np.ndarray)
        return action.astype(np.float64)

    def get_reward(self, t: float, x: np.ndarray, u: np.ndarray):
        """報酬の取得"""
        return 0.0

    def get_terminated(self, t: float, x: np.ndarray, u: np.ndarray):
        """終了条件による終了判定結果の取得"""
        return False

    def get_truncated(self, t: float, x: np.ndarray, u: np.ndarray):
        """制限時間による終了判定結果の取得"""
        return self.t >= self.t_max

    def motion_equation(self, t: float, x: np.ndarray, u: np.ndarray):
        """運動方程式"""
        raise NotImplementedError()

    def compute_energy(self, t: float, x: np.ndarray) -> float:
        """全エネルギーの計算"""
        raise NotImplementedError()

    def reset(
        self,
        initial_t: float,
        initial_x: list[float] | np.ndarray,
        integral_method: str = "runge_kutta_method",
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, bool | float | np.ndarray]]:
        """シミュレーションの初期化"""
        self.integral_method = integral_method
        self.initial_t = initial_t
        self.initial_x = np.array(initial_x, dtype=np.float64)
        self.t = initial_t
        self.x = initial_x.copy()
        s = self.get_state(self.x)
        obs = self.get_observation(self.t, self.x)
        e = self.compute_energy(self.t, self.x)
        info = {"t": self.t, "x": self.x.copy(), "e": e, "s": s.copy()}
        return obs, info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | float | np.ndarray]]:
        """シミュレーションの実行 (1ステップ)"""
        u = self.get_control_input(action)
        self.t, self.x = self.integral(self.t, self.x, u)
        s = self.get_state(self.x)
        obs = self.get_observation(self.t, self.x, u=u)
        reward = self.get_reward(self.t, self.x, u)
        terminated = self.get_terminated(self.t, self.x, u)
        truncated = self.get_truncated(self.t, self.x, u)
        e = self.compute_energy(self.t, self.x)
        info = {"t": self.t, "x": self.x.copy(), "u": u.copy(), "e": e, "s": s.copy()}
        return obs, reward, terminated, truncated, info

    def render(self) -> list[np.ndarray]:
        """図の描画"""
        raise NotImplementedError()

    @property
    def unwrapped(self) -> Env:
        return self


class MultiBodyEnv(Env):
    """マルチボディシステム用環境クラス"""

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

    def get_coordinate_indices(self) -> list[int]:
        """一般座標のインデックスの取得"""
        raise NotImplementedError()

    def get_independent_coordinate_indices(self) -> list[int]:
        """独立一般座標のインデックスの取得"""
        raise NotImplementedError()

    def get_dependent_coordinate_indices(self) -> list[int]:
        """従属一般座標のインデックスの取得"""
        pos_indices = self.get_coordinate_indices()
        independent_pos_indices = self.get_independent_coordinate_indices()
        return list(set(pos_indices) - set(independent_pos_indices))

    def get_velocity_indices(self) -> list[int]:
        """一般速度のインデックスの取得"""
        n_coordinates = len(self.get_coordinate_indices())
        return list(range(n_coordinates, 2 * n_coordinates))

    def get_independent_velocity_indices(self) -> list[int]:
        """独立一般速度のインデックスの取得"""
        n_coordinates = len(self.get_coordinate_indices())
        return [n_coordinates + idx for idx in self.get_independent_coordinate_indices()]

    def get_dependent_velocity_indices(self) -> list[int]:
        """従属一般座標のインデックスの取得"""
        vel_indices = self.get_velocity_indices()
        independent_vel_indices = self.get_independent_velocity_indices()
        return list(set(vel_indices) - set(independent_vel_indices))

    def get_state_indices(self) -> list[int]:
        """状態変数のインデックスの取得"""
        return self.get_independent_coordinate_indices() + self.get_independent_velocity_indices()

    def get_state(self, x: np.ndarray) -> np.ndarray:
        """状態の取得"""
        state_indices = self.get_state_indices()
        obs = x[state_indices].astype(self.state_space.dtype)
        return obs

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
        initial_x: list[float] | np.ndarray,
        integral_method: str = "runge_kutta_method",
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, bool | float | np.ndarray]]:
        """シミュレーションの初期化"""
        _, info = super().reset(initial_t, initial_x, integral_method=integral_method)
        self.x = self.newton_raphson_method(self.t, self.x)
        s = self.get_state(self.x)
        obs = self.get_observation(self.t, self.x)
        e = self.compute_energy(self.t, self.x)
        C = self.compute_C(self.t, self.x)
        info |= {"x": self.x.copy(), "e": e, "C": C.copy(), "s": s.copy()}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | float | np.ndarray]]:
        """シミュレーションの実行 (1ステップ)"""
        u = self.get_control_input(action)
        _, reward, terminated, truncated, info = super().step(action)
        self.x = self.newton_raphson_method(self.t, self.x)
        s = self.get_state(self.x)
        obs = self.get_observation(self.t, self.x, u=u)
        e = self.compute_energy(self.t, self.x)
        C = self.compute_C(self.t, self.x)
        info |= {"x": self.x.copy(), "e": e, "C": C.copy(), "s": s.copy()}
        return obs, reward, terminated, truncated, info

    @property
    def unwrapped(self) -> MultiBodyEnv:
        return self


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

    @property
    def x_space(self) -> Box:
        return Box(-np.inf, np.inf, (8,), np.float64)

    @property
    def u_space(self) -> Box:
        return Box(-np.inf, np.inf, (1,), np.float64)

    @property
    def state_space(self) -> Box:
        return Box(-np.inf, np.inf, (4,), np.float64)

    @property
    def observation_space(self) -> Box:
        return Box(-np.inf, np.inf, (4,), np.float64)

    @property
    def action_space(self) -> Box:
        return Box(-np.inf, np.inf, (1,), np.float64)

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

    def compute_external_force(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
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

    def get_coordinate_indices(self) -> list[int]:
        return list(range(4))

    def get_independent_coordinate_indices(self) -> list[int]:
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

    def render(self) -> list[np.ndarray]:
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        self.ax.cla()
        self.ax.set_xlim(-2.5 * self.l_pole, 2.5 * self.l_pole)
        self.ax.set_ylim(-1.5 * self.l_pole, 1.5 * self.l_pole)
        # self.ax.set_axis_off()
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
        cart_x = np.linspace(-0.15 * self.l_pole + cart_center_x, 0.15 * self.l_pole + cart_center_x, 100)
        cart_y_upper = 0.1 * self.l_pole * np.ones_like(cart_x, dtype=np.float64)
        cart_y_lower = -0.1 * self.l_pole * np.ones_like(cart_x, dtype=np.float64)
        self.ax.fill_between(cart_x, cart_y_upper, cart_y_lower, color="blue", zorder=1)

        # 車輪の描画
        wheel_right_x = cart_center_x + self.l_pole * (0.1 + 0.05 * np.cos(angle_upper))
        wheel_left_x = cart_center_x + self.l_pole * (-0.1 + 0.05 * np.cos(angle_upper))
        wheel_upper_y = self.l_pole * (-0.15 + 0.05 * np.sin(angle_upper))
        wheel_lower_y = self.l_pole * (-0.15 + 0.05 * np.sin(angle_lower))
        self.ax.fill_between(wheel_right_x, wheel_upper_y, wheel_lower_y, color="black", zorder=2)
        self.ax.fill_between(wheel_left_x, wheel_upper_y, wheel_lower_y, color="black", zorder=2)

        # 球の描画
        ball_x = ball_center[0] + 0.1 * self.l_pole * np.cos(angle_upper)
        ball_upper_y = ball_center[1] + 0.1 * self.l_pole * np.sin(angle_upper)
        ball_lower_y = ball_center[1] + 0.1 * self.l_pole * np.sin(angle_lower)
        self.ax.fill_between(ball_x, ball_upper_y, ball_lower_y, color="red", zorder=3)

        # 棒の描画
        R = np.array(
            [[np.cos(theta_pole), -np.sin(theta_pole)], [np.sin(theta_pole), np.cos(theta_pole)]], dtype=np.float64
        )
        initial_pole_upper_x = np.array([0.0, 0.0], dtype=np.float64)
        initial_pole_lower_x = np.array([-self.l_pole, 0.0], dtype=np.float64)
        pole_upper_x, pole_upper_y = ball_center + R @ initial_pole_upper_x
        pole_lower_x, pole_lower_y = ball_center + R @ initial_pole_lower_x
        self.ax.plot([pole_upper_x, pole_lower_x], [pole_upper_y, pole_lower_y], color="black", linewidth=2, zorder=2)

        # 図の描画
        self.fig.canvas.draw()
        frame = np.array(self.fig.canvas.buffer_rgba())[:, :, :3]
        return [frame]
