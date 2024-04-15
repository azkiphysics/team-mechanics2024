import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


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


if __name__ == "__main__":
    loaddir = "result"
    loadfile = "trajectory.pickle"
    loadpath = os.path.join(loaddir, loadfile)
    with open(loadpath, "rb") as f:
        result = pickle.load(f)

    t = result["t"]
    x = np.vstack(result["x"])

    fig, ax = plt.subplots()
    ax.plot(t, x[:, 0], label="$x_1$")
    ax.plot(t, x[:, 1], label="$v_1$")
    ax.plot(t, x[:, 6], label="$\\theta_2$")
    ax.plot(t, x[:, 7], label="$\\omega_2$")
    ax.legend(loc="upper right")

    savedir = loaddir
    savefile = "trajectory.png"
    savepath = os.path.join(savedir, savefile)
    fig.savefig(savepath, dpi=300)
    plt.close()
