import argparse
import logging
import os
import pickle
from logging import Formatter, StreamHandler, getLogger
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from common.agents import Agent, DDPGAgent, DQNAgent, LQRAgent, TD3Agent
from common.buffers import Buffer
from common.envs import CartPoleEnv, Env
from common.utils import FigureMaker, MovieMaker
from common.wrappers import (
    ContinuousRLMultiBodyEnvWrapper,
    DQNMultiBodyEnvWrapper,
    LQRMultiBodyEnvWrapper,
    RLCartPoleObservationWrapper,
    RLTimeObservationWrapper,
)
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

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
                    k_episodes += 1
                    logger.info(f"episode: {k_episodes}/ total_rewards: {total_rewards}")
                    self.run_result.push({"episode": k_episodes, "total_rewards": total_rewards})
                    total_rewards = 0.0
                    obs, _ = self.env.reset(**self.env_config["reset"])
                obs = next_obs.copy()

    def evaluate(self, is_render: bool = True, moviefreq: int = 1, renderfreq: int = 1):
        # シミュレーションの実行
        obs, info = self.env.reset(**self.env_config["reset"])
        self.agent.reset(**self.agent_config["reset"], is_evaluate=True)
        done = False
        self.evaluate_result.reset()
        self.evaluate_result.push(info)
        if is_render:
            self.movie_maker.reset()
            frames = self.env.render()
            self.movie_maker.add(frames[::moviefreq])
            k_movie_steps = len(frames)
            k_render_steps = 0
        while not done:
            action = self.agent.act(obs)
            next_obs, _, terminated, truncated, info = self.env.step(action)
            self.evaluate_result.push(info)
            done = terminated or truncated
            if is_render:
                k_render_steps += 1
                if k_render_steps % renderfreq == 0:
                    frames = self.env.render()
                    if moviefreq == 1:
                        self.movie_maker.add(frames)
                    else:
                        self.movie_maker.add(frames[moviefreq - k_movie_steps % moviefreq :: moviefreq])
                    k_movie_steps += len(frames)
            if done:
                break
            obs = next_obs.copy()

    def save(self, savedir: str):
        self.agent.save(savedir)
        if len(self.run_result) > 0:
            # 学習結果の保存
            training_data = self.run_result.get()
            training_savedir = os.path.join(savedir, "run")
            os.makedirs(training_savedir, exist_ok=True)
            with open(os.path.join(training_savedir, "training_data.pickle"), "wb") as f:
                pickle.dump(training_data, f)
            # 学習結果の描画
            total_rewards_data = {
                "x": {"label": "Episode", "value": np.array(training_data["episode"], dtype=np.float64)},
                "y": {
                    "label": "Total rewards",
                    "value": np.array(training_data["total_rewards"], dtype=np.float64),
                },
            }
            self.figure_maker.reset()
            self.figure_maker.make(total_rewards_data)
            savefile = "total_rewards.png"
            self.figure_maker.save(training_savedir, savefile)

        if len(self.evaluate_result) > 0:
            # 評価結果の保存
            evaluate_data = self.evaluate_result.get()
            evaluate_savedir = os.path.join(savedir, "evaluate")
            os.makedirs(evaluate_savedir, exist_ok=True)
            with open(os.path.join(evaluate_savedir, "evaluate_data.pickle"), "wb") as f:
                pickle.dump(evaluate_data, f)
            # 状態の描画
            state_data = {
                "x": {"label": "Time $t$ s", "value": np.array(evaluate_data["t"], dtype=np.float64)},
                "y": {
                    "label": "State $s$",
                    "value": [
                        {"label": "$" + f"s_{idx + 1}" + "$", "value": state_idx}
                        for idx, state_idx in enumerate(np.array(evaluate_data["s"], dtype=np.float64).T)
                    ],
                },
            }
            savefile = "state.png"
            self.figure_maker.reset()
            self.figure_maker.make(state_data)
            self.figure_maker.save(evaluate_savedir, savefile=savefile)

            # 制御入力の描画
            u_data = {
                "x": {"label": "Time $t$ s", "value": np.array(evaluate_data["t"], dtype=np.float64)[:-1]},
                "y": {
                    "label": "Control input $u$",
                    "value": [
                        {"label": "$" + f"u_{idx + 1}" + "$", "value": u_idx}
                        for idx, u_idx in enumerate(np.array(evaluate_data["u"], dtype=np.float64).T)
                    ],
                },
            }
            savefile = "u.png"
            self.figure_maker.reset()
            self.figure_maker.make(u_data)
            self.figure_maker.save(evaluate_savedir, savefile=savefile)

            # 動画の保存
            savefile = "animation.mp4"
            self.movie_maker.make(evaluate_savedir, evaluate_data["t"][-1], savefile=savefile)

    def close(self):
        self.buffer.clear()
        self.run_result.clear()
        self.evaluate_result.clear()
        self.figure_maker.close()
        self.movie_maker.close()


ENVS = {"CartPoleEnv": CartPoleEnv}
AGENTS = {"DQN": DQNAgent, "DDPG": DDPGAgent, "LQR": LQRAgent, "TD3": TD3Agent}
BUFFERS = {"Buffer": Buffer}
RUNNERS = {"Runner": Runner}
WRAPPERS = {
    "DQNMultiBody": DQNMultiBodyEnvWrapper,
    "ContinuousRLMultiBody": ContinuousRLMultiBodyEnvWrapper,
    "LQRMultiBody": LQRMultiBodyEnvWrapper,
    "RLCartPoleObservation": RLCartPoleObservationWrapper,
    "RLTimeObservation": RLTimeObservationWrapper,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Optimal control")
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    # configの読み込み
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config_savedir = config["runner"]["save"]["savedir"]
    os.makedirs(config_savedir, exist_ok=True)
    with open(os.path.join(config_savedir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    # 環境の設定
    env_config = {
        "class": ENVS[config["env"]["name"]],
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
