import glob
import os
import pickle

import numpy as np
import yaml
from tqdm import tqdm

from answer1 import AGENTS, BUFFERS, ENVS, WRAPPERS, Runner
from common.utils import FigureMaker

if __name__ == "__main__":
    _savedir = os.path.join("results", "CartPoleEnv", "Balance", "TD3", "scratch")
    savedirs = glob.glob(os.path.join(_savedir, "Q_*_R_*_Qf_*"))

    # パレート最適解集合の計算
    sse_states = []
    sse_us = []
    for savedir in tqdm(savedirs):
        config_path = os.path.join(savedir, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # 環境の設定
        env_config = {
            "class": ENVS[config["env"]["name"]],
            "wrappers": [
                {"class": WRAPPERS[wrapper["name"]], "init": wrapper["init"]} for wrapper in config["env"]["wrappers"]
            ],
            "init": config["env"]["init"],
            "reset": config["env"]["reset"],
        }

        # エージェントの設定
        loaddir = os.path.join(savedir, "agent")
        agent_config = {
            "class": AGENTS[config["agent"]["name"]],
            "init": config["agent"]["init"] | {"loaddir": loaddir},
            "reset": config["agent"]["reset"],
        }

        # バッファの設定
        buffer_config = {
            "class": BUFFERS[config["buffer"]["name"]],
            "init": config["buffer"]["init"],
            "reset": config["buffer"]["reset"],
        }

        # Runnerの設定
        config["runner"]["evaluate"]["is_render"] = False

        # Runnerの設定
        runner = Runner(env_config, agent_config, buffer_config, **config["runner"]["init"])
        runner.reset(**config["runner"]["reset"])

        # パレート最適解の計算
        runner.evaluate(**config["runner"]["evaluate"])
        t_interval = runner.evaluate_result.get()["t"][1] - runner.evaluate_result.get()["t"][0]
        states = np.array(runner.evaluate_result.get()["s"], dtype=np.float64)
        us = np.array(runner.evaluate_result.get()["u"], dtype=np.float64)
        sse_state = np.sum(np.linalg.norm(states[:-1], axis=1) ** 2) * t_interval + np.linalg.norm(states[-1]) ** 2
        sse_u = np.sum(np.linalg.norm(us, axis=1) ** 2) * t_interval

        # 結果の追加
        sse_states.append(sse_state)
        sse_us.append(sse_u)

        # Close runner
        runner.close()
    sse_states = np.array(sse_states, dtype=np.float64)
    sse_us = np.array(sse_us, dtype=np.float64)

    # データの保存
    savedir = os.path.join(_savedir, "pareto_optimal_solutions")
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "sum_square_errors.pickle"), "wb") as f:
        pickle.dump({"s": sse_states, "u": sse_us}, f)
    figure_data = {
        "x": {
            "label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{x}^T\\boldsymbol{x}dt"
            + " + \\boldsymbol{x}^T(t_f)\\boldsymbol{x}(t_f)$",
            "value": sse_states,
        },
        "y": {"label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{u}^T\\boldsymbol{u}dt$", "value": sse_us},
    }
    figure_maker = FigureMaker()
    figure_maker.reset()
    figure_maker.make(figure_data, draw_type="scatter")
    figure_maker.save(savedir, savefile="pareto_optimal_solutions.png")
    figure_maker.close()
