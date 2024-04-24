import os
from typing import Dict, List, Literal, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class Box(object):
    def __init__(
        self,
        low: float | Sequence | np.ndarray,
        high: float | Sequence | np.ndarray,
        shape: Sequence[int],
        dtype: np.float32 | np.float64,
    ) -> None:
        if isinstance(low, Sequence):
            low = np.array(low, dtype=dtype)
        if isinstance(high, Sequence):
            high = np.array(high, dtype=dtype)
        self.low = low * np.ones(shape, dtype=dtype)
        self.high = high * np.ones(shape, dtype=dtype)
        self.shape = shape
        self.dtype = dtype


class Discrete(object):
    def __init__(self, n: int) -> None:
        self.n = n


class FigureMaker(object):
    """図作成クラス"""

    def __init__(self) -> None:
        self.fig: Figure = None
        self.ax: Axes = None

    def reset(self):
        """図作成ツールの初期化"""
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
        else:
            self.ax.cla()

    def make(
        self,
        data: Dict[str, Dict[str, np.ndarray | List[Dict[str, np.ndarray]]]],
        draw_type: Literal["plot", "scatter"] = "plot",
    ):
        """図の作成"""
        x = data.get("x")
        y = data.get("y")
        x_label = x.get("label", "")
        x_value = x.get("value")
        y_label = y.get("label", "")
        y_value = y.get("value")

        if isinstance(y_value, list):
            for y_value_idx in y_value:
                if draw_type == "plot":
                    self.ax.plot(x_value, y_value_idx.get("value"), label=y_value_idx.get("label", ""))
                elif draw_type == "scatter":
                    self.ax.scatter(x_value, y_value_idx.get("value"), label=y_value_idx.get("label", ""))
            self.ax.legend(loc="upper right")
        else:
            if draw_type == "plot":
                self.ax.plot(x_value, y_value)
            elif draw_type == "scatter":
                self.ax.scatter(x_value, y_value)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

    def save(self, savedir: str, savefile: str = "trajectory.png"):
        """図の保存"""
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(savedir, savefile)
        self.fig.savefig(savepath, dpi=300)

    def close(self):
        """matplotlibを閉じる"""
        plt.close("all")


class MovieMaker(object):
    """動画作成クラス"""

    def __init__(self) -> None:
        self.frames: List[np.ndarray] = None

    def reset(self):
        self.frames = []

    def add(self, frame: np.ndarray):
        self.frames.append(frame)

    def make(self, savedir: str, t_max: float, savefile: str = "animation.mp4"):
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(savedir, savefile)
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        size = self.frames[0].shape[:2][::-1]
        video = cv2.VideoWriter(savepath, fourcc, int(len(self.frames) / t_max), size)
        for frame in self.frames:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)
        video.release()

    def close(self):
        self.frames.clear()
