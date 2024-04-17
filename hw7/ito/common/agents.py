import numpy as np

from .buffers import Buffer
from .envs import Env
from .wrappers import Wrapper, LQRMultiBodyEnvWrapper, QMultiBodyEnvWrapper


class Agent(object):
    def __init__(self, env: Env | Wrapper) -> None:
        self.env = env

    def reset(self):
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def train(self, buffer: Buffer, *args, **kwargs):
        pass


class ZeroAgent(Agent):
    def __init__(self, env: Env | Wrapper, *args, **kwargs) -> None:
        super().__init__(env)

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.action_space.shape[0], dtype=self.env.action_space.dtype)
        return action


class LQRAgent(Agent):
    def __init__(self, env: LQRMultiBodyEnvWrapper) -> None:
        self.env = env

        self.Q = None
        self.R = None

    def reset(self, Q: float | np.ndarray, R: float | np.ndarray):
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
        self.is_greedy: bool = None
        self.k_timesteps: int = None
        self.decay: float = None
        self.gamma: float = None
        self.n_samples: int = None

    def reset(
        self,
        is_greedy: bool = True,
        decay: float = 0.01,
        learning_rate: float = 0.9,
        gamma: float = 0.99,
        n_samples: int = 10,
    ):
        self.is_greedy = is_greedy
        self.k_timesteps = 0
        self.decay = decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_samples = n_samples

    def act(self, obs: np.ndarray):
        eps = 0.5 * np.exp(-self.decay * self.k_timesteps)
        if not self.is_greedy and np.random.random() < eps:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[obs])

    def train(self, buffer: Buffer, *args, **kwargs):
        data = buffer.sample(self.n_samples)
        obs = np.array(data["obs"], dtype=np.int64)
        action = np.array(data["action"], dtype=np.int64).reshape(-1, 1)
        next_obs = np.array(data["next_obs"], dtype=np.int64).reshape(-1, 1)
        reward = np.array(data["reward"], dtype=np.float64).resahpe(-1, 1)
        done = np.array(data["done"], dtype=np.int64).resahpe(-1, 1)

        q = np.take_along_axis(self.q_table[obs], action, axis=1)
        next_q = np.max(self.q_table[next_obs], axis=1, keepdims=True)
        target_q = reward + self.gamma * (1.0 - done) * next_q
        td_error = target_q - q
        one_hot_action = np.identity(self.env.action_space.n)[action.reshape(-1)]
        self.q_table[obs] += self.learning_rate * one_hot_action * td_error
