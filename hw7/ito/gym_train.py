import argparse
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.agents import Agent, DQNAgent, DDPGAgent
from common.buffers import Buffer
from common.envs import Env, CartPoleEnv
from common.utils import MovieMaker
from common.wrappers import DQNMultiBodyEnvWrapper, DDPGMultiBodyEnvWrapper

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


AGENTS = {"DQN": DQNAgent, "DDPG": DDPGAgent}
ENVS = {"CartPole": CartPoleEnv}
ENV_WRAPPERS = {
    "DQNMultiBody": DQNMultiBodyEnvWrapper,
    "DDPGMultiBody": DDPGMultiBodyEnvWrapper,
}


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
        if "wrappers" in env_config:
            for env_wrapper_config in self.env_config["wrappers"]:
                self.env = env_wrapper_config["class"](self.env, **env_wrapper_config["init"])
        self.agent: Agent = agent_config["class"](self.env, **agent_config["init"])
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

    def run(self, total_timesteps: int, learning_starts: int = 1000, trainfreq: int | None = None) -> Dict[str, float]:
        k_episodes = 0
        total_rewards = 0.0
        obs, _ = self.env.reset(**self.env_config["reset"])
        for k_timesteps in tqdm(range(total_timesteps)):
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
            if (
                k_timesteps >= learning_starts
                and trainfreq is not None
                and trainfreq > 0
                and k_timesteps % trainfreq == 0
            ):
                self.agent.train(buffer=self.buffer)
            if done:
                print("total_rewards: ", total_rewards)
                k_episodes = 0
                total_rewards = 0.0
                obs, _ = self.env.reset(**self.env_config["reset"])
            obs = next_obs.copy()
            self.run_result.push({"episode": k_episodes, "total_rewards": total_rewards})


if __name__ == "__main__":
    args = argparse.ArgumentParser("gym env training")
    parser = args.parse_args()

    # 環境の設定
    env_name = "Pendulum-v1"
    agent_name = "DDPG"
    env_config = {
        "class": gym.make,
        "wrappers": [],
        "init": {"id": env_name},
        "reset": {},
    }

    # エージェントの設定
    agent_config = {"class": AGENTS[agent_name], "init": {}, "reset": {}}

    # バッファの設定
    buffer_config = {"class": Buffer, "init": {"maxlen": None}, "reset": {}}

    # Runnerの設定
    runner = Runner(env_config, agent_config, buffer_config)
    runner.reset()
    # runner.run(2)
    runner.run(200000, learning_starts=10000, trainfreq=1)
