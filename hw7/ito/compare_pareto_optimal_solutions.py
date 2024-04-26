import os
import pickle

import matplotlib.pyplot as plt

from common.utils import FigureMaker


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
    savedir = os.path.join("results", "CartPoleEnv")
    max_actions = [10.0, 15.0, 25.0]
    subdirs = [
        "LQR",
        *[os.path.join("TD3", f"action_{idx}") for idx in max_actions],
    ]
    savepaths = [
        os.path.join(savedir, subdir, "pareto_optimal_solutions", "sum_square_errors.pickle") for subdir in subdirs
    ]

    figure_maker = FigureMaker()
    figure_maker.reset()
    for idx, savepath in enumerate(savepaths):
        with open(savepath, "rb") as f:
            if idx == 0:
                label = "LQR"
            else:
                label = "RL (maximum of $\\boldsymbol{u}$: " + f"{max_actions[idx - 1]})"
            data = pickle.load(f)
            figure_data = {
                "x": {
                    "label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{x}^T\\boldsymbol{x}dt"
                    + " + \\boldsymbol{x}^T(t_f)\\boldsymbol{x}(t_f)$",
                    "value": data["s"],
                },
                "y": {
                    "label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{u}^T\\boldsymbol{u}dt$",
                    "value": [{"label": label, "value": data["u"]}],
                },
            }
            figure_maker.make(figure_data, draw_type="scatter")
    figure_maker.save(savedir, "pareto_optimal_solutions.png")
