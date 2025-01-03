import os
import pickle
import random
from collections import defaultdict, deque

import numpy as np


class Buffer(object):
    """データ格納クラス"""

    def __init__(self, maxlen: int | None = None) -> None:
        self.maxlen = maxlen

        self.buffer: dict[str, list[float | np.ndarray]] | None = None
        self.n_data: int | None = None

    def reset(self):
        self.buffer = defaultdict(lambda: deque(maxlen=self.maxlen))
        self.n_data = 0

    def push(self, data: dict[str, list[float | np.ndarray]]):
        if self.buffer is None:
            self.reset()
        for key, value in data.items():
            self.buffer[key].append(value)
        self.n_data += 1
        if self.maxlen is not None:
            self.n_data = min(self.n_data, self.maxlen)

    def sample(self, n_samples: int) -> dict[str, list[float | np.ndarray]]:
        if self.n_data == 0:
            return self.buffer
        n_samples = min(self.n_data, n_samples)
        indices = random.sample(range(self.n_data), n_samples)
        data = {key: [value[idx] for idx in indices] for key, value in self.buffer.items()}
        return data

    def get(self) -> dict[str, list[float | np.ndarray]]:
        return {key: value for key, value in self.buffer.items()}

    def clear(self):
        self.buffer.clear()

    def save(self, savedir: str, savefile: str):
        buffer = self.get()
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, savefile)
        with open(path, "wb") as f:
            pickle.dump(buffer, f)

    def __len__(self):
        return self.n_data
