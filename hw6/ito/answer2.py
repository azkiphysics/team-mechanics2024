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
plt.rcParams["axes.axisbelow"] = True  # グリッドを最背面に移動


if __name__ == "__main__":
    loaddir = savedir = "result"
    fig, ax = plt.subplots(figsize=(8, 6))
    for integral_method in ["euler_method", "runge_kutta_method"]:
        savepath = os.path.join(loaddir, integral_method, "trajectory.pickle")
        with open(savepath, "rb") as f:
            result = pickle.load(f)
        total_energy_mae = np.mean(np.abs(np.array(result["e"]) - result["e"][0]))
        total_calc_speed = np.sum(result["calc_speed"])
        ax.scatter([total_energy_mae], [total_calc_speed], label=integral_method.replace("_", " "))
    ax.set_xlabel("Mean absolute error (MAE) of total energy (J)")
    ax.set_ylabel("Total calculation speed (s)")
    ax.legend(loc="upper right")
    savepath = os.path.join(savedir, "compare_integral_method.png")
    fig.savefig(savepath, dpi=300)
    plt.close()
