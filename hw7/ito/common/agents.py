import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffers import Buffer
from .envs import Env
from .wrappers import (
    Wrapper,
    LQRMultiBodyEnvWrapper,
    QMultiBodyEnvWrapper,
    DQNMultiBodyEnvWrapper,
    DDPGMultiBodyEnvWrapper,
)


class Agent(object):
    def __init__(self, env: Env | Wrapper) -> None:
        self.env = env

    def reset(self, **kwargs):
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train(self, buffer: Buffer, **kwargs):
        pass


class ZeroAgent(Agent):
    def __init__(self, env: Env | Wrapper, **kwargs) -> None:
        super().__init__(env)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.action_space.shape[0], dtype=self.env.action_space.dtype)
        return action


class LQRAgent(Agent):
    def __init__(self, env: LQRMultiBodyEnvWrapper) -> None:
        self.env = env

        self.Q = None
        self.R = None

    def reset(self, Q: float | np.ndarray, R: float | np.ndarray, **kwargs):
        A = self.env.A
        B = self.env.B
        self.Q = Q * np.identity(self.env.observation_space.shape[0], dtype=self.env.observation_space.dtype)
        self.R = R * np.identity(self.env.action_space.shape[0], dtype=self.env.action_space.dtype)
        AH = np.block([[A, -B @ np.linalg.inv(self.R) @ B.T], [-self.Q, -A.T]])
        eig_values, eig_vectors = np.linalg.eig(AH)
        S1, S2 = np.split(eig_vectors[:, eig_values < 0], 2, axis=0)
        P = S2 @ np.linalg.inv(S1)
        self.K = (-np.linalg.inv(self.R) @ B.T @ P).real

    def act(self, obs: np.ndarray) -> np.ndarray:
        return (self.K @ obs).astype(self.env.action_space.dtype)


class QAgent(Agent):
    def __init__(self, env: QMultiBodyEnvWrapper) -> None:
        self.env = env

        self.q_table = np.random.uniform(low=-1, high=1, size=(self.env.observation_space.n, self.env.action_space.n))
        self.k_timesteps: int = None
        self.is_evaluate: bool = None
        self.eps_low: float = None
        self.initial_eps: float = None
        self.decay: float = None
        self.gamma: float = None
        self.batch_size: int = None

    def reset(
        self,
        eps_low: float = 0.01,
        initial_eps: float = 1.0,
        decay: float = 0.01,
        learning_rate: float = 0.5,
        gamma: float = 0.99,
        batch_size: int = 1,
        eps_update_freq: int = 100,
        is_evaluate: bool = False,
        **kwargs,
    ):
        self.eps_low = eps_low
        self.initial_eps = initial_eps
        self.k_timesteps = 0
        self.k_decay = 0
        self.decay = decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_update_freq = eps_update_freq
        self.is_evaluate = is_evaluate

    def act(self, obs: np.ndarray) -> np.int64:
        self.k_timesteps += 1
        if self.k_timesteps % self.eps_update_freq == 0:
            self.k_decay += 1
        eps = self.eps_low + (self.initial_eps - self.eps_low) * np.exp(-self.decay * self.k_decay)
        if not self.is_evaluate and np.random.random() < eps:
            return np.array(np.random.randint(self.env.action_space.n), dtype=np.int64)
        else:
            return np.argmax(self.q_table[obs])

    def train(self, buffer: Buffer, **kwargs):
        data = buffer.sample(self.batch_size)
        obs = np.array(data["obs"], dtype=np.int64)
        action = np.array(data["action"], dtype=np.int64).reshape(-1)
        next_obs = np.array(data["next_obs"], dtype=np.int64)
        reward = np.array(data["reward"], dtype=np.float64).reshape(-1, 1)
        done = np.array(data["done"], dtype=np.int64).reshape(-1, 1)

        one_hot_action = np.identity(self.env.action_space.n)[action]
        q = self.q_table[obs]
        next_q = np.max(self.q_table[next_obs], axis=1, keepdims=True)
        target_q = reward + self.gamma * (1.0 - done) * next_q
        td_error = target_q - q
        self.q_table[obs] += self.learning_rate * one_hot_action * td_error


class DRLAgent(Agent):
    def get_device(self, device: torch.device | str = "auto") -> torch.device:
        """
        Retrieve PyTorch device.
        It checks that the requested device is available first.
        For now, it supports only cpu and cuda.
        By default, it tries to use the gpu.

        :param device: One for 'auto', 'cuda', 'cpu'
        :return: Supported Pytorch device
        """
        # Cuda by default
        if device == "auto":
            device = "cuda"
        # Force conversion to torch.device
        device = torch.device(device)

        # Cuda not available
        if device.type == torch.device("cuda").type and not torch.cuda.is_available():
            return torch.device("cpu")

        return device


class DQNAgent(DRLAgent):
    def __init__(
        self,
        env: DQNMultiBodyEnvWrapper,
        eps_low: float = 0.01,
        initial_eps: float = 1.0,
        decay: float = 0.01,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        eps_update_freq: int = 10,
        tau: float = 1.0,
        target_update_interval: int = 10000,
        max_grad_norm: float = 10.0,
        device: torch.device | str = "auto",
    ) -> None:
        self.env = env

        n_observations = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.device = self.get_device(device=device)
        self.q_network = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        ).to(self.device)

        self.target_q_network = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.train(False)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.eps_low = eps_low
        self.initial_eps = initial_eps
        self.decay = decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_update_freq = eps_update_freq
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm

        self.k_timesteps: int = None
        self.is_evaluate: bool = None
        self.k_trains: int = None
        self.k_decay: int = None

    def reset(self, is_evaluate: bool = False, **kwargs):
        self.k_timesteps = 0
        self.k_decay = 0
        self.k_trains = 0
        self.is_evaluate = is_evaluate

    def act(self, obs: np.ndarray) -> np.int64:
        self.k_timesteps += 1
        if self.k_timesteps % self.eps_update_freq == 0:
            self.k_decay += 1
        eps = self.eps_low + (self.initial_eps - self.eps_low) * np.exp(-self.decay * self.k_decay)
        if not self.is_evaluate and np.random.random() < eps:
            return np.array(np.random.randint(self.env.action_space.n), dtype=np.int64)
        else:
            obs_pt = torch.tensor(
                obs.reshape(-1, self.env.observation_space.shape[0]), dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                q_pt = self.q_network.forward(obs_pt)
            q = q_pt.to("cpu").detach().numpy()[0]
            return np.argmax(q)

    def train(self, buffer: Buffer, **kwargs):
        self.k_trains += 1
        self.q_network.train(True)

        # サンプルデータをtorch.tensorに変換
        data = buffer.sample(self.batch_size)
        obs_pt = torch.tensor(np.array(data["obs"], dtype=np.float32), dtype=torch.float32, device=self.device)
        action = np.array(data["action"], dtype=np.int64).reshape(-1)
        one_hot_action_pt = torch.tensor(
            np.identity(self.env.action_space.n)[action], dtype=torch.float32, device=self.device
        )
        next_obs_pt = torch.tensor(
            np.array(data["next_obs"], dtype=np.float32), dtype=torch.float32, device=self.device
        )
        reward_pt = torch.tensor(
            np.array(data["reward"], dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        done_pt = torch.tensor(
            np.array(data["done"], dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device
        )

        # Q-networkの更新
        q_pt = torch.sum(one_hot_action_pt * self.q_network(obs_pt), dim=1, keepdim=True)
        with torch.no_grad():
            next_q_pt, _ = torch.max(self.target_q_network(next_obs_pt), dim=1, keepdim=True)
            target_q_pt = reward_pt + self.gamma * (1.0 - done_pt) * next_q_pt
        # loss_pt = 0.5 * F.mse_loss(q_pt, target_q_pt)
        loss_pt = 0.5 * F.smooth_l1_loss(q_pt, target_q_pt)
        self.optimizer.zero_grad()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        loss_pt.backward()
        self.optimizer.step()

        if self.k_trains % self.target_update_interval == 0:
            # Target networkの更新 (Polyak update)
            with torch.no_grad():
                for param, target_param in zip(
                    self.q_network.parameters(),
                    self.target_q_network.parameters(),
                ):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)

        self.q_network.train(False)


class DDPGAgent(DRLAgent):
    def __init__(
        self,
        env: DDPGMultiBodyEnvWrapper,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 256,
        tau: float = 0.005,
        action_noise: float = 0.1,
        device: torch.device | str = "auto",
    ) -> None:
        self.env = env
        n_observations = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
        self.device = self.get_device(device=device)

        self.policy = nn.Sequential(
            nn.Linear(n_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh(),
        ).to(self.device)
        self.target_policy = nn.Sequential(
            nn.Linear(n_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh(),
        ).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.train(False)

        self.q_network = nn.Sequential(
            nn.Linear(n_observations + n_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        ).to(self.device)
        self.target_q_network = nn.Sequential(
            nn.Linear(n_observations + n_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.train(False)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=critic_learning_rate)

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.action_noise = action_noise

        self.is_evaluate: bool = None

    def reset(self, is_evaluate: bool = False, **kwargs):
        self.is_evaluate = is_evaluate

    def act(self, obs: np.ndarray):
        obs_pt = torch.tensor(
            obs.reshape(-1, self.env.observation_space.shape[0]), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            scaled_action_pt: torch.Tensor = self.policy.forward(obs_pt)
        scaled_action: np.ndarray = scaled_action_pt.to("cpu").detach().numpy()[0]
        if self.is_evaluate:
            noise = np.zeros_like(scaled_action, dtype=np.float32)
        else:
            noise = np.random.normal(0.0, self.action_noise, scaled_action.shape[0]).astype(np.float32)

        clipped_ratio = (1.0 + np.clip(scaled_action + noise, -1.0, 1.0)) / 2.0
        low = self.env.action_space.low
        high = self.env.action_space.high
        clipped_action = low + (high - low) * clipped_ratio
        return clipped_action

    def train(self, buffer: Buffer, **kwargs):
        low_pt = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
        high_pt = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)

        # サンプルデータをtorch.tensorに変換
        data = buffer.sample(self.batch_size)
        obs_pt = torch.tensor(np.array(data["obs"], dtype=np.float32), dtype=torch.float32, device=self.device)
        action_pt = -1.0 + 2.0 * (
            torch.tensor(np.array(data["action"], dtype=np.float32), dtype=torch.float32, device=self.device) - low_pt
        ) / (high_pt - low_pt)
        next_obs_pt = torch.tensor(
            np.array(data["next_obs"], dtype=np.float32), dtype=torch.float32, device=self.device
        )
        reward_pt = torch.tensor(
            np.array(data["reward"], dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        done_pt = torch.tensor(
            np.array(data["done"], dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device
        )

        # Q-networkの更新
        self.q_network.train(True)
        obs_action_pt = torch.cat([obs_pt, action_pt], dim=1)
        q_pt = self.q_network.forward(obs_action_pt)
        with torch.no_grad():
            target_next_action_pt = self.target_policy.forward(next_obs_pt).clamp(-1, 1)
            next_obs_target_next_action_pt = torch.cat([next_obs_pt, target_next_action_pt], dim=1)
            target_next_q_pt = self.target_q_network(next_obs_target_next_action_pt)
            target_q_pt = reward_pt + self.gamma * (1.0 - done_pt) * target_next_q_pt
        critic_loss_pt = F.mse_loss(q_pt, target_q_pt)
        self.critic_optimizer.zero_grad()
        critic_loss_pt.backward()
        self.critic_optimizer.step()
        self.q_network.train(False)

        # Policyの更新
        self.policy.train(True)
        action_pi_pt = self.policy.forward(obs_pt).clamp(-1, 1)
        obs_action_pi_pt = torch.concat([obs_pt, action_pi_pt], dim=1)
        q_pi_pt = self.q_network.forward(obs_action_pi_pt)
        actor_loss_pt = -q_pi_pt.mean()
        self.actor_optimizer.zero_grad()
        actor_loss_pt.backward()
        self.actor_optimizer.step()
        self.policy.train(False)

        # Target networkの更新 (Polyak update)
        with torch.no_grad():
            for param, target_param in zip(
                self.q_network.parameters(),
                self.target_q_network.parameters(),
            ):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)
            for param, target_param in zip(
                self.policy.parameters(),
                self.target_policy.parameters(),
            ):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)
