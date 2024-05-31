import argparse
import logging
from logging import Formatter, StreamHandler, getLogger

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from common.agents import Agent, DDPGAgent, DQNAgent, TD3Agent
from common.buffers import Buffer
from common.utils import FigureMaker, MovieMaker

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
        env_config: dict[str, gym.Env | dict[str, int | float] | list[dict[str, int | float]]],
        agent_config: dict[str, Agent | dict[str, int | float]],
        buffer_config: dict[str, Buffer | dict[str, int]],
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
        self.evaluate_result = Buffer()
        self.figure_maker = FigureMaker()
        self.movie_maker = MovieMaker()

    def reset(self):
        self.env.reset(**self.env_config["reset"])
        self.agent.reset(**self.agent_config["reset"])
        self.buffer.reset(**self.buffer_config["reset"])
        self.run_result.reset()
        self.evaluate_result.reset()
        self.figure_maker.reset()
        self.movie_maker.reset()

    def run(self, total_timesteps: int, learning_starts: int = 1000, trainfreq: int | None = None) -> dict[str, float]:
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

    def evaluate(self, moviefreq: int = 100):
        # シミュレーションの実行
        obs, _ = self.env.reset(**self.env_config["reset"])
        self.agent.reset(**self.agent_config["reset"], is_evaluate=True)
        done = False
        k_steps = 0
        self.evaluate_result.reset()
        self.evaluate_result.push({"step": k_steps, "obs": obs.copy()})
        self.movie_maker.reset()
        self.movie_maker.add(self.env.render())
        while not done:
            k_steps += 1
            action = self.agent.act(obs)
            next_obs, _, terminated, truncated, _ = self.env.step(action)
            self.evaluate_result.push({"step": k_steps, "obs": next_obs.copy(), "action": action.copy()})
            done = terminated or truncated
            if k_steps % moviefreq == 0 or done:
                self.movie_maker.add(self.env.render())
            if done:
                break
            obs = next_obs.copy()

    def save(self, savedir: str):
        if len(self.run_result) > 0:
            # 学習結果の保存
            training_data = self.run_result.get()
            total_rewards_data = {
                "x": {"label": "Episode", "value": np.array(training_data["episode"], dtype=np.float64)},
                "y": {
                    "label": "State $s$",
                    "value": np.array(training_data["total_rewards"], dtype=np.float64),
                },
            }
            self.figure_maker.reset()
            self.figure_maker.make(total_rewards_data)
            savefile = "total_rewards.png"
            self.figure_maker.save(savedir, savefile)

        if len(self.evaluate_result) > 0:
            evaluate_data = self.evaluate_result.get()
            # 状態の保存
            obs_data = {
                "x": {"label": "Step", "value": np.array(evaluate_data["step"], dtype=np.float64)},
                "y": {
                    "label": "Observation $o$",
                    "value": [
                        {"label": "$" + f"o_{idx + 1}" + "$", "value": state_idx}
                        for idx, state_idx in enumerate(np.array(evaluate_data["obs"], dtype=np.float64).T)
                    ],
                },
            }
            savefile = "observation.png"
            self.figure_maker.reset()
            self.figure_maker.make(obs_data)
            self.figure_maker.save(savedir, savefile=savefile)

            # 制御入力の保存
            action = np.array(evaluate_data["action"], dtype=np.float64)
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                action = action.reshape(-1, 1)
            action_data = {
                "x": {"label": "Step", "value": np.array(evaluate_data["step"], dtype=np.float64)[:-1]},
                "y": {
                    "label": "Action $a$",
                    "value": [
                        {"label": "$" + f"a_{idx + 1}" + "$", "value": u_idx} for idx, u_idx in enumerate(action.T)
                    ],
                },
            }
            savefile = "action.png"
            self.figure_maker.reset()
            self.figure_maker.make(action_data)
            self.figure_maker.save(savedir, savefile=savefile)

            # 動画の保存
            savefile = "animation.mp4"
            self.movie_maker.make(savedir, 10.0, savefile=savefile)


AGENTS = {"DQN": DQNAgent, "DDPG": DDPGAgent, "TD3": TD3Agent}
BUFFERS = {"Buffer": Buffer}
RUNNERS = {"Runner": Runner}
WRAPPERS = {}

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
        "wrappers": [
            {"class": WRAPPERS[wrapper["name"]], "init": wrapper["init"]} for wrapper in config["env"]["wrappers"]
        ],
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
    runner.save(**config["runner"]["save"])
