import os
import pickle

import matplotlib.pyplot as plt

from common.utils import FigureMaker3d

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
    savedir = os.path.join("results", "CartPoleEnv", "Balance")
    subdirs = ["LQR", "TD3/scratch"]
    savepaths = [
        os.path.join(savedir, subdir, "pareto_optimal_solutions", "sum_square_errors.pickle") for subdir in subdirs
    ]
    labels = ["LQR", "TD3"]

    figure_maker = FigureMaker3d()
    figure_maker.reset()
    figure_data = []
    for label, savepath in zip(labels, savepaths):
        with open(savepath, "rb") as f:
            data = pickle.load(f)
            figure_data_idx = {
                "label": label,
                "x": {
                    "label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{x}^T\\boldsymbol{x}dt$",
                    "value": data["s"],
                },
                "y": {
                    "label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{u}^T\\boldsymbol{u}dt$",
                    "value": data["u"],
                },
                "z": {"label": "$\\boldsymbol{x}^T(t_f)\\boldsymbol{x}(t_f)$", "value": data["final_s"]},
            }
            figure_data.append(figure_data_idx)
    figure_maker.make(figure_data)
    figure_maker.save(savedir, "pareto_optimal_solutions.png")
