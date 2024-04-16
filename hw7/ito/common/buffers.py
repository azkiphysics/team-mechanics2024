import os
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np


class Buffer(object):
    """データ格納クラス"""

    def __init__(self) -> None:
        self.buffer: Dict[str, List[float | np.ndarray]] | None = None

    def reset(self):
        self.buffer = defaultdict(list)

    def push(self, data: Dict[str, List[float | np.ndarray]]):
        if self.buffer is None:
            self.reset()
        for key, value in data.items():
            self.buffer[key].append(value)

    def get(self) -> Dict[str, List[float | np.ndarray]] | None:
        return self.buffer

    def clear(self):
        self.buffer.clear()

    def save(self, savedir: str, savefile: str):
        buffer = self.get()
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, savefile)
        with open(path, "wb") as f:
            pickle.dump(buffer, f)
