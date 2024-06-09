import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


# Matplotlibで綺麗な論文用のグラフを作る
# https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 15 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 15 # 軸だけ変更されます
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid


if __name__ == "__main__":
    loaddir = "result"
    loadfile = "trajectory.pickle"
    loadpath = os.path.join(loaddir, loadfile)
    with open(loadpath, "rb") as f:
        result = pickle.load(f)

    t = result["t"]
    x, y, z = np.split(np.vstack(result["x"]), 3, axis=1)

    fig = plt.figure(figsize=(9, 6), layout="constrained")
    axs = fig.subplot_mosaic(
        [
            ["trajectory", "time_series_x"],
            ["trajectory", "time_series_y"],
            ["trajectory", "time_series_z"],
        ],
        per_subplot_kw={("trajectory",): {"projection": "3d"}},
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.15, "hspace": 0.05},
    )
    axs["trajectory"].plot(x, y, z)
    axs["trajectory"].set_xlabel("x")
    axs["trajectory"].set_ylabel("y")
    axs["trajectory"].set_zlabel("z")

    axs["time_series_x"].plot(t, x)
    axs["time_series_x"].set_xlabel("Time $t$ s")
    axs["time_series_x"].set_ylabel("$x$")

    axs["time_series_y"].plot(t, y)
    axs["time_series_y"].set_xlabel("Time $t$ s")
    axs["time_series_y"].set_ylabel("$y$")

    axs["time_series_z"].plot(t, z)
    axs["time_series_z"].set_xlabel("Time $t$ s")
    axs["time_series_z"].set_ylabel("$z$")

    savedir = loaddir
    savefile = "trajectory.png"
    savepath = os.path.join(savedir, savefile)
    fig.savefig(savepath, dpi=300)
    plt.close()