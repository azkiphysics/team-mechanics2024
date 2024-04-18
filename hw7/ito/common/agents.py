import numpy as np

from .buffers import Buffer
from .envs import Env
from .wrappers import Wrapper, LQRMultiBodyEnvWrapper, QMultiBodyEnvWrapper


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

    def act(self, obs: np.ndarray):
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
        self.n_batches: int = None

    def reset(
        self,
        eps_low: float = 0.01,
        initial_eps: float = 1.0,
        decay: float = 0.01,
        learning_rate: float = 0.5,
        gamma: float = 0.99,
        n_batches: int = 1,
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
        self.n_batches = n_batches
        self.eps_update_freq = eps_update_freq
        self.is_evaluate = is_evaluate

    def act(self, obs: np.ndarray):
        self.k_timesteps += 1
        if self.k_timesteps % self.eps_update_freq == 0:
            self.k_decay += 1
        eps = self.eps_low + (self.initial_eps - self.eps_low) * np.exp(-self.decay * self.k_decay)
        if not self.is_evaluate and np.random.random() < eps:
            return np.array(np.random.randint(self.env.action_space.n), dtype=np.int64)
        else:
            return np.argmax(self.q_table[obs])

    def train(self, buffer: Buffer, **kwargs):
        data = buffer.sample(self.n_batches)
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
