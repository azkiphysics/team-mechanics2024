import argparse
from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from common.agents import Agent, ZeroAgent
from common.buffers import Buffer
from common.envs import Env, CartPoleEnv

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


class Runner(object):
    def __init__(
        self,
        env_cls: Env,
        env_config: Dict[str, Dict[str, int | float]],
        agent_cls: Agent,
        agent_config: Dict[str, Dict[str, int | float]],
    ) -> None:
        self.env_cls = env_cls
        self.env_config = env_config
        self.agent_cls = agent_cls
        self.agent_config = agent_config

        self.env: Env = env_cls(**env_config["init"])
        self.agent: Agent = agent_cls(self.env.observation_space, self.env.action_space, **agent_config["init"])
        self.buffer = Buffer()

    def reset(self):
        self.env.reset(**self.env_config["reset"])
        self.agent.reset(**self.agent_config["reset"])
        self.buffer.reset()

    def run(self, n_episodes: int, train_freq: int = 1) -> Dict[str, float]:
        k_timesteps = 0
        results = defaultdict(list)
        for k_episodes in range(n_episodes):
            k_steps = 0
            total_rewards = 0.0
            obs = self.env.reset(**self.env_init_config)
            while True:
                k_timesteps += 1
                k_steps += 1
                action = self.agent.act(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.push(
                    {
                        "obs": obs.copy(),
                        "next_obs": next_obs.copy(),
                        "action": action.copy(),
                        "reward": reward,
                        "done": done,
                    }
                )
                total_rewards += reward
                if k_timesteps % train_freq == 0:
                    self.agent.train(buffer=self.buffer)
                if done:
                    break
                obs = next_obs.copy()
            results["episode"].append(k_episodes)
            results["total_rewards"].append(total_rewards)
        return results


if __name__ == "__main__":
    args = argparse.ArgumentParser("Cart pole problem")
    args.add_argument("integral_method", choices=["euler_method", "runge_kutta_method"])
    parser = args.parse_args()

    # 環境の設定
    t_max = 10.0
    dt = 1e-3
    m_cart = 1.0
    m_ball = 1.0
    l_pole = 1.0
    initial_t = 0.0
    initial_x = np.array([0.0, 0.0, 1.0, np.pi / 2 - 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    integral_method = "runge_kutta_method"
    env_cls = CartPoleEnv
    env_init_config = dict(t_max=t_max, dt=dt, m_cart=m_cart, m_ball=m_ball, l_pole=l_pole)
    env_reset_config = dict(initial_t=initial_t, initial_x=initial_x)

    # エージェントの設定
    agent_cls = ZeroAgent
    agent_init_config = {}
    agent_reset_config = {}

    # Runnerの設定
    runner = Runner(env_cls, env_init_config, env_reset_config, agent_cls, agent_init_config, agent_reset_config)
    runner.reset()
