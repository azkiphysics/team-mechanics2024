import datetime
import logging
import os
from collections import deque
from logging import Formatter, StreamHandler, getLogger
from typing import Any, Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env, ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers.record_video import RecordVideo

# ロガーの設定
if __name__ == "__main__":
    logger_level = logging.INFO
else:
    logger_level = logging.WARNING
logger = getLogger(__name__)
logger.setLevel(logger_level)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = StreamHandler()
handler.setLevel(logger_level)
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


class QLearning(object):
    """Q学習エージェント

    本クラスはε-greedy法を利用したQ学習エージェントです.
    Qテーブルは各ステップの遷移データを利用して更新します.
    observation_spaceとaction_spaceはいずれも離散空間である必要があります.
    """

    def __init__(
        self,
        observation_space: Discrete,
        action_space: Discrete,
        alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon_start: float = 0.5,
    ) -> None:
        """エージェントの初期化

        Args:
            observation_space (Discrete): 環境クラスの観測空間
            action_space (Discrete): 環境クラスの行動空間
            alpha (float): 学習率
            epsilon_start (float): ε-greedy法のεの初期値
        """

        self.observation_space = observation_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start

        n_observations = self.observation_space.n
        n_actions = self.action_space.n
        self.q_table = np.random.uniform(-1, 1, size=(n_observations, n_actions))

        self.epsilon: float = None
        self.episode: int = None

    def _compute_epsilon(self, episode: int) -> float:
        """εの計算

        Args:
            episode (int): εの更新回数

        Returns:
            float: εの計算結果
        """
        return self.epsilon_start / (episode + 1)

    def reset(self) -> None:
        """エージェントの学習の初期化"""
        self.episode = 0
        self.epsilon = self._compute_epsilon(self.episode)

    def get_action(self, obs: int, deterministic: bool = False, update_epsilon: bool = False) -> int:
        """エージェントの行動選択

        Args:
            obs (int): 環境から受け取る観測結果
            deterministic (bool, optional): greedyに行動選択するかどうか Defaults to False.
            update_epsilon (bool, optional): εの値を更新するかどうか Defaults to False.

        Returns:
            int: 行動の値
        """
        if update_epsilon:
            self.episode += 1
            self.epsilon = self._compute_epsilon(self.episode)
        if deterministic or self.epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[obs])
        else:
            action = self.action_space.sample()
        return action

    def update(
        self, data: dict[Literal["obs", "next_obs", "action", "reward", "done"], int | bool]
    ) -> dict[Literal["loss"], float]:
        """Q-tableの更新

        Args:
            data: (dict[Literal["obs", "next_obs", "action", "reward", "done"], int | bool]): 遷移データ

        Returns:
            dict[Literal["loss"], float]: loss関数 (TD誤差の2乗)
        """
        q = self.q_table[data["obs"], data["action"]]
        next_q_max = np.max(self.q_table[data["next_obs"]])
        td_error = data["reward"] + self.gamma * (1.0 - data["done"]) * next_q_max - q
        self.q_table[data["obs"], data["action"]] += self.alpha * td_error
        return {"loss": 0.5 * td_error**2}

    def save(self, savedir: str) -> None:
        """Q-tableの保存

        Args:
            savedir (str): Q-tableを保存するディレクトリ
        """
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, "q_table.csv")
        np.savetxt(path, self.q_table, delimiter=",")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(self.q_table, aspect="auto")
        ax.set_xlabel("Action")
        ax.set_ylabel("Observation")
        n_x_ticks = min(10, self.action_space.n)
        xticks = range(0, self.action_space.n, (self.action_space.n + n_x_ticks - 1) // n_x_ticks)
        n_y_ticks = min(10, self.observation_space.n)
        yticks = range(0, self.observation_space.n, (self.observation_space.n + n_y_ticks - 1) // n_y_ticks)
        ax.set_xticks(xticks, map(lambda x: str(x), xticks))
        ax.set_yticks(yticks, map(lambda x: str(x), yticks))
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Q value")
        path = os.path.join(savedir, "q_table.png")
        fig.savefig(path, dpi=300)
        plt.close()


class DigitizedObservationWrapper(ObservationWrapper):
    """観測量を離散化するObservationWrapper

    本クラスは, 環境クラス(Env)から受け取る観測量を離散化するための環境ラッパーです．
    env.resetとenv.stepの戻り値の一つである観測量obs/next_obsを連続値から整数値に変換します．
    """

    def __init__(self, env: Env, obs_min: tuple[float, ...], obs_max: tuple[float, ...], n_digitized: int) -> None:
        """観測量を離散化するObservationWrapperの初期化

        Args:
            env (Env): 環境
            obs_min (tuple[float, ...]): 離散化前の観測量の最小値
            obs_max (tuple[float, ...]): 離散化前の観測量の最大値
            n_digitized (int): 離散化の分割数
        """
        assert isinstance(env.unwrapped.observation_space, Box)
        super().__init__(env)
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.n_digitized = n_digitized

    @property
    def observation_space(self) -> Discrete:
        """状態空間

        Returns:
            Discrete: 離散化した状態空間
        """
        return Discrete(self.n_digitized ** self.unwrapped.observation_space.shape[0])

    def _generate_bins(self, idx: int) -> np.ndarray:
        """i番目の観測量の離散化用の箱の生成

        Args:
            idx (int): 環境(self.env)の観測量(obs/ next_obs)のインデックス

        Returns:
            np.ndarray: 環境(self.env)のi番目の観測量の離散化用のn等分された箱
        """
        return np.linspace(self.obs_min[idx], self.obs_max[idx], self.n_digitized + 1)[1:-1]

    def observation(self, observation: np.ndarray) -> int:
        """離散化された観測量の計算

        Args:
            observation (np.ndarray): 離散化前の観測量

        Returns:
            int: 離散化後の観測量
        """
        return sum(
            [
                np.digitize(obs_idx, bins=self._generate_bins(idx)) * (self.n_digitized**idx)
                for idx, obs_idx in enumerate(observation)
            ]
        )


class CartPoleWrapper(Wrapper):
    """倒立振子環境用の環境ラッパー

    本クラスはCartPole-v1で使用する環境ラッパーです.
    最大ステップ数と報酬をカスタムします．
    """

    def __init__(self, env: Env, n_steps: int):
        """倒立振子環境用の環境ラッパーの初期化

        Args:
            env (Env): 環境
            n_steps (int): 1エピソードの最大ステップ数
        """
        super().__init__(env)
        self.n_steps = n_steps

        self.k_steps: int = None

    def _compute_truncated(self) -> bool:
        """truncatedの計算

        Returns:
            bool: 現在のステップが最大ステップかどうか
        """
        return self.k_steps >= self.n_steps

    def _compute_reward(self, done: bool) -> float:
        """報酬の計算

        Args:
            done: 1エピソードが終了したかどうか

        Returns:
            float: カスタム報酬
        """
        if done:
            if self.k_steps < 195:
                reward = -200
            else:
                reward = 1
        else:
            reward = 1
        return reward

    def step(self, action: np.ndarray) -> tuple[int | np.ndarray, float, bool, bool, dict[str, Any]]:
        """環境のステップの更新

        Args:
            action (np.ndarray): エージェントの行動

        Returns:
            int | np.ndarray: 観測量
            float: 報酬
            bool: 最大ステップ数までに倒立維持が失敗したかどうか
            bool: 最大ステップ後も倒立維持しているかどうか
            dict[str, Any]: ステップを更新したときの情報
        """
        self.k_steps += 1
        next_obs, _, terminated, _, info = super().step(action)
        truncated = self._compute_truncated()
        reward = self._compute_reward(terminated or truncated)
        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[int | np.ndarray, dict[str, Any]]:
        """シミュレーションの初期化

        Returns:
            int | np.ndarray: 観測量
            dict[str, Any]: シミュレーションを初期化したときの情報
        """
        self.k_steps = 0
        obs, info = super().reset(**kwargs)
        return obs, info


class Runner(object):
    """学習の実行クラス

    本クラスは, 環境とエージェントを利用してシミュレーションを行い,エージェントの学習を行います.
    """

    def __init__(self, env: Env, agent: QLearning) -> None:
        """Runnerクラスの初期化

        Args:
            env (Env): 環境
            agent (QLearning): Q学習エージェント
        """
        self.env = env
        self.agent = agent

    def reset(self) -> None:
        """エージェントの初期化"""
        self.agent.reset()

    def run(
        self,
        n_episodes: int,
        n_evaluate_episodes: int,
        goal_average_total_reward: float,
        savedir: str,
        savefreq: int = -1,
    ) -> dict[Literal["episode", "total_reward", "loss"], list[float], list[tuple[float, float]]]:
        """エージェントの学習

        Args:
            n_episodes (int): シミュレーションのエピソード数
            n_evaluate_episodes (int): early stoppingするかの判定に利用する直近のエピソードの個数
            goal_average_total_reward (float): 累積報酬和の目標値
            savedir (str): 保存ディレクトリ
            savefreq (int, optional): 保存頻度 (-1の場合は最終エピソードのみ保存) Defaults to -1.

        Returns:
            dict[Literal["", "total_reward", "loss"], list[float], list[tuple[float, float]]]: 学習結果 (累積報酬和と損失関数)
        """
        total_rewards = []
        total_rewards_past_n_episodes = deque(maxlen=n_evaluate_episodes)
        losses = []
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            k_steps = 0
            total_reward = 0.0
            losses_per_episode = []
            while True:
                update_epsilon = episode > 0 and k_steps == 0
                action = self.agent.get_action(obs, update_epsilon=update_epsilon)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                data = {"obs": obs, "next_obs": next_obs, "action": action, "reward": reward, "done": terminated}
                learning_result = self.agent.update(data)

                total_reward += reward
                losses_per_episode.append(learning_result["loss"])

                if terminated or truncated:
                    loss_mean = np.mean(losses_per_episode)
                    loss_std = np.std(losses_per_episode)
                    logger.info(
                        f"Episode: {episode} / Total reward: {total_reward}"
                        + f" / Loss (Mean): {loss_mean} / Loss (Std): {loss_std}"
                    )
                    break
                else:
                    k_steps += 1
                    obs = next_obs
            total_rewards_past_n_episodes.append(total_reward)
            total_rewards.append(total_reward)
            losses.append((loss_mean, loss_std))

            if (
                (savefreq > 0 and episode % savefreq == 0)
                or (savefreq == -1 and episode == n_episodes - 1)
                or (savefreq != 0 and np.mean(total_rewards_past_n_episodes) >= goal_average_total_reward)
            ):
                evaluate_savedir = os.path.join(savedir, "evaluate", f"episode_{episode}")
                self.evaluate(evaluate_savedir)

            if np.mean(total_rewards_past_n_episodes) >= goal_average_total_reward:
                logger.info(f"Successful over the past {n_evaluate_episodes} episodes!")
                break
        return {"episode": np.arange(len(total_rewards)), "total_reward": total_rewards, "loss": losses}

    def evaluate(self, savedir: str) -> None:
        """エージェントの評価

        Args:
            savedir (str): 保存ディレクトリ
        """
        env = RecordVideo(self.env, savedir)
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            action = self.agent.get_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            else:
                obs = next_obs
        env.close()
        self.agent.save(savedir)


if __name__ == "__main__":
    # 倒立振子環境の設定
    obs_min = (-2.4, -3.0, -0.5, -2.0)
    obs_max = (2.4, 3.0, 0.5, 2.0)
    n_digitized = 6
    n_steps = 200
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = DigitizedObservationWrapper(env, obs_min, obs_max, n_digitized)
    env = CartPoleWrapper(env, n_steps)

    # Q学習の設定
    alpha = 0.5
    gamma = 0.99
    epsilon_start = 0.5
    agent = QLearning(env.observation_space, env.action_space, alpha=alpha, gamma=gamma, epsilon_start=epsilon_start)

    # シミュレーションの設定
    n_evaluate_episodes = 100
    n_episodes = 2000
    goal_average_reward = 195
    savefreq = 100
    savedir = os.path.join("results", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    runner = Runner(env, agent)

    # シミュレーションの実行
    runner.reset()
    results = runner.run(n_episodes, n_evaluate_episodes, goal_average_reward, savedir, savefreq=savefreq)

    # 学習結果の整形
    episodes = results["episode"]
    total_rewards = results["total_reward"]
    losses = results["loss"]
    losses_mean, losses_std = list(map(lambda x: np.array(x, dtype=np.float64), list(zip(*losses))))
    _savedir = os.path.join(savedir, "run")
    os.makedirs(_savedir, exist_ok=True)

    # 学習データの保存
    for key, value in results.items():
        savefile = key + ".csv"
        savepath = os.path.join(_savedir, savefile)
        np.savetxt(savepath, value, delimiter=",")

    # 累積報酬和の図示
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(episodes, total_rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    savepath = os.path.join(_savedir, "total_reward.png")
    fig.savefig(savepath, dpi=300)
    plt.close()

    # 損失関数の図示
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(episodes, losses_mean)
    ax.fill_between(episodes, losses_mean + losses_std, np.maximum(0.0, losses_mean - losses_std), alpha=0.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    savepath = os.path.join(_savedir, "loss.png")
    fig.savefig(savepath, dpi=300)
    plt.close()
