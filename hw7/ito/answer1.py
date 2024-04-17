import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from common.agents import Agent, ZeroAgent
from common.buffers import Buffer
from common.envs import Env, CartPoleEnv
from common.utils import MovieMaker
from common.wrappers import LQRMultiBodyEnvWrapper

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
        env_config: Dict[str, Env | Dict[str, int | float] | List[Dict[str, int | float]]],
        agent_config: Dict[str, Agent | Dict[str, int | float]],
        buffer_config: Dict[str, Buffer | Dict[str, int]],
    ) -> None:
        self.env_config = env_config
        self.agent_config = agent_config
        self.buffer_config = buffer_config

        self.env: Env = env_config["class"](**env_config["init"])
        if "wrapper" in env_config:
            for env_wrapper_config in self.env_config["wrapper"]:
                self.env = env_wrapper_config["class"](self.env, **env_wrapper_config["init"])
        self.agent: Agent = agent_config["class"](
            self.env.observation_space, self.env.action_space, **agent_config["init"]
        )
        self.buffer: Buffer = buffer_config["class"](**buffer_config["init"])
        self.run_result = Buffer()
        self.evaluate_result = Buffer()
        self.movie_maker = MovieMaker()

    def reset(self):
        self.env.reset(**self.env_config["reset"])
        self.agent.reset(**self.agent_config["reset"])
        self.buffer.reset(**self.buffer_config["reset"])
        self.run_result.reset()
        self.evaluate_result.reset()

    def run(self, n_episodes: int, train_freq: int | None = None) -> Dict[str, float]:
        k_timesteps = 0
        for k_episodes in range(n_episodes):
            k_steps = 0
            total_rewards = 0.0
            obs, _ = self.env.reset(**self.env_config["reset"])
            self.agent.reset(**self.agent_config["reset"])
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
                if train_freq is not None and train_freq > 0 and k_timesteps % train_freq == 0:
                    self.agent.train(buffer=self.buffer)
                if done:
                    break
                obs = next_obs.copy()
            self.run_result.push({"episode": k_episodes, "total_rewards": total_rewards})

    def evaluate(self, savedir: str):
        # シミュレーションの実行
        obs, info = self.env.reset(**self.env_config["reset"])
        self.agent.reset(**self.agent_config["reset"])
        done = False
        self.evaluate_result.reset()
        self.evaluate_result.push(info)
        self.movie_maker.reset()
        self.movie_maker.add(self.env.render())
        k_steps = 0
        movie_freq = 100
        while not done:
            k_steps += 1
            action = self.agent.act(obs)
            next_obs, _, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.evaluate_result.push(info)
            if k_steps % movie_freq == 0 or done:
                self.movie_maker.add(self.env.render())
            if done:
                break
            obs = next_obs.copy()

        # 結果の保存
        savefile = "evaluate_result.pickle"
        self.evaluate_result.save(savedir, savefile)

        # 動画の保存
        savefile = "evaluate_result.mp4"
        self.movie_maker.make(savedir, self.env.unwrapped.t_max, savefile=savefile)


if __name__ == "__main__":
    args = argparse.ArgumentParser("Cart pole")
    parser = args.parse_args()

    # 環境の設定
    initial_x = np.array([0.0, 0.0, 1.0, np.pi / 2 - 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    target_x = initial_x.copy()
    target_x[0] = 1.0
    target_x[3] = np.pi / 2.0
    env_config = {
        "class": CartPoleEnv,
        "wrapper": [{"class": LQRMultiBodyEnvWrapper, "init": {}}],
        "init": {"t_max": 10.0, "dt": 1e-3, "m_cart": 1.0, "m_ball": 1.0, "l_pole": 1.0},
        "reset": {
            "initial_t": 0.0,
            "initial_x": initial_x,
            "integral_method": "runge_kutta_method",
            "target_x": target_x,
            "Q": 1.0,
            "R": 1.0,
        },
    }

    # エージェントの設定
    agent_config = {"class": ZeroAgent, "init": {}, "reset": {}}

    # バッファの設定
    buffer_config = {"class": Buffer, "init": {"maxlen": 1000}, "reset": {}}

    # Runnerの設定
    runner = Runner(env_config, agent_config, buffer_config)
    runner.reset()
    runner.run(1)
    runner.evaluate("result")
