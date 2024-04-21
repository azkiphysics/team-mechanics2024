import argparse
import logging
from logging import getLogger, Formatter, StreamHandler
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
import yaml
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from common.agents import Agent, DQNAgent, DDPGAgent
from common.buffers import Buffer
from common.utils import MovieMaker

# ロガーの設定
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(handler_format)
logger.addHandler(handler)

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
        env_config: Dict[str, gym.Env | Dict[str, int | float] | List[Dict[str, int | float]]],
        agent_config: Dict[str, Agent | Dict[str, int | float]],
        buffer_config: Dict[str, Buffer | Dict[str, int]],
    ) -> None:
        self.env_config = env_config
        self.agent_config = agent_config
        self.buffer_config = buffer_config

        self.env: gym.Env = env_config["class"](**env_config["init"])
        if "wrappers" in env_config:
            for env_wrapper_config in self.env_config["wrappers"]:
                self.env = env_wrapper_config["class"](self.env, **env_wrapper_config["init"])
        self.agent: Agent = agent_config["class"](self.env, **agent_config["init"])
        self.buffer: Buffer = buffer_config["class"](**buffer_config["init"])
        self.run_result = Buffer()
        self.movie_maker = MovieMaker()

    def reset(self):
        self.env.reset(**self.env_config["reset"])
        self.agent.reset(**self.agent_config["reset"])
        self.buffer.reset(**self.buffer_config["reset"])
        self.run_result.reset()
        self.movie_maker.reset()

    def run(self, total_timesteps: int, learning_starts: int = 1000, trainfreq: int | None = None) -> Dict[str, float]:
        k_episodes = 0
        total_rewards = 0.0
        obs, _ = self.env.reset(**self.env_config["reset"])
        with logging_redirect_tqdm(loggers=[logger]):
            for k_timesteps in trange(total_timesteps, leave=False):
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
                    logger.info(f"episode: {k_episodes}/ total_rewards: {total_rewards}")
                    k_episodes += 1
                    total_rewards = 0.0
                    obs, _ = self.env.reset(**self.env_config["reset"])
                obs = next_obs.copy()
                self.run_result.push({"episode": k_episodes, "total_rewards": total_rewards})

    def evaluate(self, savedir: str, movie_freq: int = 100):
        # シミュレーションの実行
        obs, _ = self.env.reset(**self.env_config["reset"])
        self.agent.reset(**self.agent_config["reset"], is_evaluate=True)
        done = False
        self.movie_maker.reset()
        self.movie_maker.add(self.env.render())
        k_steps = 0
        while not done:
            k_steps += 1
            action = self.agent.act(obs)
            next_obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            if k_steps % movie_freq == 0 or done:
                self.movie_maker.add(self.env.render())
            if done:
                break
            obs = next_obs.copy()

        # 動画の保存
        savefile = "evaluate_result.mp4"
        self.movie_maker.make(savedir, 10.0, savefile=savefile)


AGENTS = {"DQN": DQNAgent, "DDPG": DDPGAgent}
BUFFERS = {"Buffer": Buffer}
RUNNERS = {"Runner": Runner}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gymnasium training")
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    # configの読み込み
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    # 環境の設定
    env_config = {
        "class": gym.make,
        "wrappers": config["env"]["wrappers"],
        "init": config["env"]["init"],
        "reset": config["env"]["reset"],
    }

    # エージェントの設定
    agent_config = {
        "class": AGENTS[config["agent"]["name"]],
        "init": config["agent"]["init"],
        "reset": config["agent"]["reset"],
    }

    # バッファの設定
    buffer_config = {
        "class": BUFFERS[config["buffer"]["name"]],
        "init": config["buffer"]["init"],
        "reset": config["buffer"]["reset"],
    }

    # Runnerの設定
    runner: Runner = RUNNERS[config["runner"]["name"]](
        env_config, agent_config, buffer_config, **config["runner"]["init"]
    )
    runner.reset(**config["runner"]["reset"])
    runner.run(**config["runner"]["run"])
    runner.evaluate(**config["runner"]["evaluate"])
