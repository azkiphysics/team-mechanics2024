from typing import Dict

import numpy as np


class Agent(object):
    def __init__(self) -> None:
        pass

    def reset(self):
        pass

    def act(self) -> np.ndarray:
        pass

    def train(self, data: Dict[str, np.ndarray]):
        pass
