import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

from .buffers import Buffer
from .envs import Env
from .policies import DDPGActor, DDPGCritic, DQNPolicy, SACActor, SACCritic
from .wrappers import (
    ContinuousRLMultiBodyEnvWrapper,
    DQNMultiBodyEnvWrapper,
    LQRMultiBodyEnvWrapper,
    Wrapper,
)


class Agent(object):
    def __init__(self, env: Env | Wrapper) -> None:
        self.env = env
        self.is_evaluate: bool = False

    def reset(self, **kwargs):
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train(self, buffer: Buffer, **kwargs):
        pass

    def save(self, savedir: str):
        pass

    def evaluate_mode(self, is_evaluate: bool):
        self.is_evaluate = is_evaluate


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

    def save(self, savedir: str):
        _savedir = os.path.join(savedir, "agent")
        os.makedirs(_savedir, exist_ok=True)
        lqr_data = {"Q": self.Q, "R": self.R, "K": self.K}
        with open(os.path.join(_savedir, "lqr_data.pickle"), "wb") as f:
            pickle.dump(lqr_data, f)


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
        target_update_interval: int = 1000,
        max_grad_norm: float = 10.0,
        scheduler_last_ratio: float = 1.0,
        scheduler_iters: int = 0,
        device: torch.device | str = "auto",
        loaddir: str | None = None,
    ) -> None:
        self.env = env
        self.is_evaluate: bool = False

        self.device = self.get_device(device=device)
        self.q_network = DQNPolicy(env.observation_space, env.action_space).to(self.device)
        self.target_q_network = DQNPolicy(env.observation_space, env.action_space).to(self.device)

        if loaddir is not None:
            q_network_path = os.path.join(loaddir, "q_network.pt")
            target_q_network_path = os.path.join(loaddir, "taget_q_network.pt")
            self.q_network.load_state_dict(torch.load(q_network_path))
            self.target_q_network.load_state_dict(torch.load(target_q_network_path))
        else:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.train(False)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = LinearLR(self.optimizer, 1.0, scheduler_last_ratio, total_iters=scheduler_iters)

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
        self.k_trains: int = None
        self.k_decay: int = None

    def reset(self, **kwargs):
        self.k_timesteps = 0
        self.k_decay = 0
        self.k_trains = 0

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
        self.scheduler.step()

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

    def save(self, savedir: str):
        _savedir = os.path.join(savedir, "agent")
        os.makedirs(_savedir, exist_ok=True)
        q_network_path = os.path.join(_savedir, "q_network.pt")
        target_q_network_path = os.path.join(_savedir, "target_q_network.pt")
        self.q_network.to("cpu")
        self.target_q_network.to("cpu")
        torch.save(self.q_network.state_dict(), q_network_path)
        torch.save(self.target_q_network.state_dict(), target_q_network_path)
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)


class DDPGAgent(DRLAgent):
    def __init__(
        self,
        env: ContinuousRLMultiBodyEnvWrapper,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 256,
        tau: float = 0.005,
        action_noise: float = 0.1,
        actor_scheduler_last_ratio: float = 1.0,
        actor_scheduler_iters: int = 0,
        critic_scheduler_last_ratio: float = 1.0,
        critic_scheduler_iters: int = 0,
        device: torch.device | str = "auto",
        loaddir: str | None = None,
    ) -> None:
        self.env = env

        self.device = self.get_device(device=device)
        self.policy = DDPGActor(env.observation_space, env.action_space).to(self.device)
        self.target_policy = DDPGActor(env.observation_space, env.action_space).to(self.device)

        self.q_network = DDPGCritic(env.observation_space, env.action_space).to(self.device)
        self.target_q_network = DDPGCritic(env.observation_space, env.action_space).to(self.device)

        if loaddir is not None:
            policy_path = os.path.join(loaddir, "policy.pt")
            target_policy_path = os.path.join(loaddir, "target_policy.pt")
            q_network_path = os.path.join(loaddir, "q_network.pt")
            target_q_network_path = os.path.join(loaddir, "target_q_network.pt")
            self.policy.load_state_dict(torch.load(policy_path))
            self.target_policy.load_state_dict(torch.load(target_policy_path))
            self.q_network.load_state_dict(torch.load(q_network_path))
            self.target_q_network.load_state_dict(torch.load(target_q_network_path))
        else:
            self.target_policy.load_state_dict(self.policy.state_dict())
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_policy.train(False)
        self.target_q_network.train(False)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_learning_rate)
        self.actor_scheduler = LinearLR(
            self.actor_optimizer, 1.0, actor_scheduler_last_ratio, total_iters=actor_scheduler_iters
        )
        self.critic_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=critic_learning_rate)
        self.critic_scheduler = LinearLR(
            self.critic_optimizer, 1.0, critic_scheduler_last_ratio, total_iters=critic_scheduler_iters
        )

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.action_noise = action_noise

        self.is_evaluate: bool = False

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
        critic_loss_pt = 0.5 * F.mse_loss(q_pt, target_q_pt)
        self.critic_optimizer.zero_grad()
        critic_loss_pt.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()
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
        self.actor_scheduler.step()
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

    def save(self, savedir: str):
        _savedir = os.path.join(savedir, "agent")
        os.makedirs(_savedir, exist_ok=True)
        policy_path = os.path.join(_savedir, "policy.pt")
        target_policy_path = os.path.join(_savedir, "target_policy.pt")
        q_network_path = os.path.join(_savedir, "q_network.pt")
        target_q_network_path = os.path.join(_savedir, "target_q_network.pt")
        self.policy.to("cpu")
        self.target_policy.to("cpu")
        self.q_network.to("cpu")
        self.target_q_network.to("cpu")
        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.target_policy.state_dict(), target_policy_path)
        torch.save(self.q_network.state_dict(), q_network_path)
        torch.save(self.target_q_network.state_dict(), target_q_network_path)
        self.policy.to(self.device)
        self.target_policy.to(self.device)
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)


class TD3Agent(DDPGAgent):
    def __init__(
        self,
        env: ContinuousRLMultiBodyEnvWrapper,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 256,
        tau: float = 0.005,
        action_noise: float = 0.1,
        actor_scheduler_last_ratio: float = 1.0,
        actor_scheduler_iters: int = 0,
        critic_scheduler_last_ratio: float = 1.0,
        critic_scheduler_iters: int = 0,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: torch.device | str = "auto",
        loaddir: str | None = None,
    ) -> None:
        self.env = env
        self.device = self.get_device(device=device)

        self.policy = DDPGActor(env.observation_space, env.action_space).to(self.device)
        self.target_policy = DDPGActor(env.observation_space, env.action_space).to(self.device)

        self.q_networks = tuple(DDPGCritic(env.observation_space, env.action_space).to(self.device) for _ in range(2))
        self.target_q_networks = tuple(
            DDPGCritic(env.observation_space, env.action_space).to(self.device) for _ in range(2)
        )
        if loaddir is not None:
            policy_path = os.path.join(loaddir, "policy.pt")
            target_policy_path = os.path.join(loaddir, "target_policy.pt")
            self.policy.load_state_dict(torch.load(policy_path))
            self.target_policy.load_state_dict(torch.load(target_policy_path))
        else:
            self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.train(False)
        for idx, (q_network, target_q_network) in enumerate(zip(self.q_networks, self.target_q_networks)):
            if loaddir is not None:
                q_network_path = os.path.join(loaddir, f"q_network{idx + 1}.pt")
                target_q_network_path = os.path.join(loaddir, f"target_q_network{idx + 1}.pt")
                q_network.load_state_dict(torch.load(q_network_path))
                target_q_network.load_state_dict(torch.load(target_q_network_path))
            else:
                target_q_network.load_state_dict(q_network.state_dict())
            target_q_network.train(False)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_learning_rate)
        self.actor_scheduler = LinearLR(
            self.actor_optimizer, 1.0, actor_scheduler_last_ratio, total_iters=actor_scheduler_iters
        )
        self.critic_optimizer = torch.optim.Adam(
            tuple(self.q_networks[0].parameters()) + tuple(self.q_networks[1].parameters()), lr=critic_learning_rate
        )
        self.critic_scheduler = LinearLR(
            self.critic_optimizer, 1.0, critic_scheduler_last_ratio, total_iters=critic_scheduler_iters
        )

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.action_noise = action_noise
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay

        self.is_evaluate: bool = False
        self._n_update: int = None

    def reset(self, **kwargs):
        self._n_update = 0

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
        self._n_update += 1
        for q_network in self.q_networks:
            q_network.train(True)
        obs_action_pt = torch.cat([obs_pt, action_pt], dim=1)
        qs_pt = tuple(q_network.forward(obs_action_pt) for q_network in self.q_networks)
        with torch.no_grad():
            target_next_action_pt = self.target_policy.forward(next_obs_pt)
            noise_pt = (
                target_next_action_pt.clone()
                .data.normal_(0.0, self.target_noise)
                .clamp(-self.target_noise_clip, self.target_noise_clip)
            )
            target_next_action_pt = (target_next_action_pt + noise_pt).clamp(-1.0, 1.0)
            next_obs_target_next_action_pt = torch.cat([next_obs_pt, target_next_action_pt], dim=1)
            target_next_q_pt, _ = torch.min(
                torch.cat(
                    tuple(
                        target_q_network(next_obs_target_next_action_pt) for target_q_network in self.target_q_networks
                    ),
                    dim=1,
                ),
                dim=1,
                keepdim=True,
            )
            target_q_pt = reward_pt + self.gamma * (1.0 - done_pt) * target_next_q_pt
        critic_loss_pt = 0.5 * sum(F.mse_loss(q_pt, target_q_pt) for q_pt in qs_pt)
        self.critic_optimizer.zero_grad()
        critic_loss_pt.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        for q_network in self.q_networks:
            q_network.train(False)

        # Policyの更新
        if self._n_update % self.policy_delay == 0:
            self.policy.train(True)
            action_pi_pt = self.policy.forward(obs_pt).clamp(-1, 1)
            obs_action_pi_pt = torch.concat([obs_pt, action_pi_pt], dim=1)
            q_pi_pt: torch.Tensor = self.q_networks[0].forward(obs_action_pi_pt)
            actor_loss_pt = -q_pi_pt.mean()
            self.actor_optimizer.zero_grad()
            actor_loss_pt.backward()
            self.actor_optimizer.step()
            self.actor_scheduler.step()
            self.policy.train(False)

            # Target networkの更新 (Polyak update)
            with torch.no_grad():
                for q_network, target_q_network in zip(self.q_networks, self.target_q_networks):
                    for param, target_param in zip(q_network.parameters(), target_q_network.parameters()):
                        target_param.data.mul_(1 - self.tau)
                        target_param.data.add_(self.tau * param.data)
                for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)

    def save(self, savedir: str):
        _savedir = os.path.join(savedir, "agent")
        os.makedirs(_savedir, exist_ok=True)
        policy_path = os.path.join(_savedir, "policy.pt")
        target_policy_path = os.path.join(_savedir, "target_policy.pt")
        self.policy.to("cpu")
        self.target_policy.to("cpu")
        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.target_policy.state_dict(), target_policy_path)
        self.policy.to(self.device)
        self.target_policy.to(self.device)
        for idx, (q_network, target_q_network) in enumerate(zip(self.q_networks, self.target_q_networks)):
            q_network_path = os.path.join(_savedir, f"q_network{idx + 1}.pt")
            target_q_network_path = os.path.join(_savedir, f"target_q_network{idx + 1}.pt")
            q_network.to("cpu")
            target_q_network.to("cpu")
            torch.save(q_network.state_dict(), q_network_path)
            torch.save(target_q_network.state_dict(), target_q_network_path)
            q_network.to(self.device)
            target_q_network.to(self.device)


class SACAgent(DRLAgent):
    def __init__(
        self,
        env: ContinuousRLMultiBodyEnvWrapper,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        ent_coef_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        tau: float = 0.005,
        actor_scheduler_last_ratio: float = 1.0,
        actor_scheduler_iters: int = 0,
        critic_scheduler_last_ratio: float = 1.0,
        critic_scheduler_iters: int = 0,
        ent_coef: str | float = "auto",
        target_entropy: str | float = "auto",
        device: torch.device | str = "auto",
        loaddir: str | None = None,
    ) -> None:
        self.env = env
        self.device = self.get_device(device=device)

        self.policy = SACActor(env.observation_space, env.action_space).to(self.device)

        self.q_networks = tuple(SACCritic(env.observation_space, env.action_space).to(self.device) for _ in range(2))
        self.target_q_networks = tuple(
            SACCritic(env.observation_space, env.action_space).to(self.device) for _ in range(2)
        )
        if loaddir is not None:
            policy_path = os.path.join(loaddir, "policy.pt")
            self.policy.load_state_dict(torch.load(policy_path))
        for idx, (q_network, target_q_network) in enumerate(zip(self.q_networks, self.target_q_networks)):
            if loaddir is not None:
                q_network_path = os.path.join(loaddir, f"q_network{idx + 1}.pt")
                target_q_network_path = os.path.join(loaddir, f"target_q_network{idx + 1}.pt")
                q_network.load_state_dict(torch.load(q_network_path))
                target_q_network.load_state_dict(torch.load(target_q_network_path))
            else:
                target_q_network.load_state_dict(q_network.state_dict())
            target_q_network.train(False)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_learning_rate)
        self.actor_scheduler = LinearLR(
            self.actor_optimizer, 1.0, actor_scheduler_last_ratio, total_iters=actor_scheduler_iters
        )
        self.critic_optimizer = torch.optim.Adam(
            tuple(self.q_networks[0].parameters()) + tuple(self.q_networks[1].parameters()), lr=critic_learning_rate
        )
        self.critic_scheduler = LinearLR(
            self.critic_optimizer, 1.0, critic_scheduler_last_ratio, total_iters=critic_scheduler_iters
        )

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.ent_coef = ent_coef

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in ent_coef:
                init_value = float(ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly
            # different from the paper as discussed in
            # https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = nn.Parameter(torch.ones(1, device=self.device) * init_value, requires_grad=True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=ent_coef_learning_rate)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string
            # (different from 'auto') is passed
            self.log_ent_coef = nn.Parameter(
                torch.log(torch.tensor(float(ent_coef), device=self.device)) + 1e-6, requires_grad=False
            )
            self.ent_coef_optimizer = None

        if target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
        else:
            self.target_entropy = float(self.target_entropy)

        self.is_evaluate: bool = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_pt = torch.tensor(
            obs.reshape(-1, self.env.observation_space.shape[0]), dtype=torch.float32, device=self.device
        )
        if self.is_evaluate:
            with torch.no_grad():
                scaled_action_pt = self.policy.get_deterministic_action(obs_pt)
        else:
            with torch.no_grad():
                scaled_action_pt = self.policy.get_probabilistic_action(obs_pt)
        scaled_action: np.ndarray = scaled_action_pt.to("cpu").detach().numpy()[0]

        clipped_ratio = (1.0 + scaled_action) / 2.0
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

        ent_coef = torch.exp(self.log_ent_coef.detach())
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            self.log_ent_coef.requires_grad_(True)
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            log_prob_pi_pt = self.policy.log_prob_pi(obs_pt)
            ent_coef_loss = -(self.log_ent_coef * (log_prob_pi_pt + self.target_entropy).detach()).mean()

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            self.log_ent_coef.requires_grad_(False)

        # Q-networkの更新
        for q_network in self.q_networks:
            q_network.train(True)
        obs_action_pt = torch.cat([obs_pt, action_pt], dim=1)
        qs_pt = tuple(q_network.forward(obs_action_pt) for q_network in self.q_networks)
        with torch.no_grad():
            next_action_pi_pt = self.policy.get_probabilistic_action(next_obs_pt)
            next_log_prob_pi_pt = self.policy.log_prob(next_obs_pt, next_action_pi_pt)
            next_obs_next_action_pi_pt = torch.cat((next_obs_pt, next_action_pi_pt), dim=1)
            target_next_q_pi_pt, _ = torch.min(
                torch.cat(
                    tuple(target_q_network(next_obs_next_action_pi_pt) for target_q_network in self.target_q_networks),
                    dim=1,
                ),
                dim=1,
                keepdim=True,
            )
            target_q_pt = reward_pt + self.gamma * (1.0 - done_pt) * (
                target_next_q_pi_pt - ent_coef * next_log_prob_pi_pt
            )
        critic_loss_pt = 0.5 * sum(F.mse_loss(q_pt, target_q_pt) for q_pt in qs_pt)
        self.critic_optimizer.zero_grad()
        critic_loss_pt.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        for q_network in self.q_networks:
            q_network.train(False)

        # Policyの更新
        self.policy.train(True)
        action_pi_pt = self.policy.get_probabilistic_action(obs_pt)
        log_prob_pi_pt = self.policy.log_prob(obs_pt, action_pi_pt)
        obs_action_pi_pt = torch.concat([obs_pt, action_pi_pt], dim=1)
        q_pi_pt, _ = torch.min(
            torch.cat(tuple(q_network(obs_action_pi_pt) for q_network in self.q_networks), dim=1), dim=1, keepdim=True
        )
        actor_loss_pt = (-q_pi_pt + ent_coef * log_prob_pi_pt).mean()
        self.actor_optimizer.zero_grad()
        actor_loss_pt.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()
        self.policy.train(False)

        # Target networkの更新 (Polyak update)
        with torch.no_grad():
            for q_network, target_q_network in zip(self.q_networks, self.target_q_networks):
                for param, target_param in zip(q_network.parameters(), target_q_network.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)

    def save(self, savedir: str):
        _savedir = os.path.join(savedir, "agent")
        os.makedirs(_savedir, exist_ok=True)
        policy_path = os.path.join(_savedir, "policy.pt")
        self.policy.to("cpu")
        torch.save(self.policy.state_dict(), policy_path)
        self.policy.to(self.device)
        for idx, (q_network, target_q_network) in enumerate(zip(self.q_networks, self.target_q_networks)):
            q_network_path = os.path.join(_savedir, f"q_network{idx + 1}.pt")
            target_q_network_path = os.path.join(_savedir, f"target_q_network{idx + 1}.pt")
            q_network.to("cpu")
            target_q_network.to("cpu")
            torch.save(q_network.state_dict(), q_network_path)
            torch.save(target_q_network.state_dict(), target_q_network_path)
            q_network.to(self.device)
            target_q_network.to(self.device)
