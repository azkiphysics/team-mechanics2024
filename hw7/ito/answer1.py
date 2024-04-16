import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from common.agents import Agent
from common.buffers import Buffer
from common.envs import MultiBodyEnv, CartPoleEnv

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


class Worker(object):
    def __init__(
        self,
        env_cls: MultiBodyEnv,
        env_init_config: Dict[str, float],
        agent_cls: Agent,
        agent_init_config: Dict[str],
    ) -> None:
        self.env: MultiBodyEnv = env_cls(**env_init_config)
        self.agent: Agent = agent_cls(self.env.observation_space, self.env.action_space, **agent_init_config)
        self.buffer = Buffer()

        self.obs = None
        self.done: bool = False

    def reset(self, env_reset_config: Dict[str, float | np.ndarray], agent_reset_config: Dict[str, float | np.ndarray]):
        info = self.env.reset(**env_reset_config)
        self.agent.reset(**agent_reset_config)
        self.buffer.reset()
        self.buffer.push(info)
        self.obs = info.get("obs")
        self.done = info.get("truncated") or info.get("terminated")

    def run(self, n_steps: int | None = None) -> Dict[str, float | np.ndarray]:
        k_steps = 0
        while True:
            k_steps += 1
            action = self.agent.act(self.obs)
            info = self.env.step(action)
            self.obs = info.get("obs")
            reward = info.get("reward")
            self.done = info.get("truncated") or info.get("terminated")
            self.buffer.push({"obs": self.obs, "action": action, "reward": reward, "done": self.done})
            if (k_steps is not None and k_steps == n_steps) or (k_steps is None and self.done):
                break
            if self.done:
                info = self.env.reset()
                self.buffer.push(info)
        result = self.buffer.get()
        self.buffer.clear()
        return result


class Runner(object):
    def __init__(self, env_cls: MultiBodyEnv, env_init_config: Dict[str, float]) -> None:
        self.worker = Worker(env_cls, env_init_config)

    def reset(self, env_reset_config: Dict[str, float | np.ndarray]):
        self.worker.reset(env_reset_config)

    def run(self, total_timesteps: int, n_steps: int | None = None):
        n_updates = total_timesteps // n_steps
        n_steps_total_updates = [n_steps] * n_updates
        if total_timesteps - n_updates * n_steps > 0:
            n_steps_total_updates += [total_timesteps - n_updates * n_steps]
        for n_steps_per_update in n_steps_total_updates:
            result = self.worker.run(n_steps=n_steps_per_update)
        return result

    def save(self, savdir: str, savefile: str):
        self.buffer


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
    initial_x = np.array([0.0, 0.0, 1.0, np.pi / 2 - 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    integral_method = "runge_kutta_method"

    env_init_config = dict(t_max=t_max, dt=dt, m_cart=m_cart, m_ball=m_ball, l_pole=l_pole)
    env_reset_config = dict(initial_t=initial_t, initial_x=initial_x)

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
    independent_x = np.array(data.get("x"), dtype=np.float64)[:, independent_state_indices]
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
