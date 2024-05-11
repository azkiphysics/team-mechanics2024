import os
import pickle

import numpy as np
import yaml
from tqdm import tqdm

from answer1 import Runner
from common.agents import LQRAgent
from common.buffers import Buffer
from common.envs import CartPoleEnv
from common.utils import FigureMaker3d
from common.wrappers import LQRMultiBodyEnvWrapper

if __name__ == "__main__":
    config_path = os.path.join("configs", "CartPoleEnv", "Balance", "LQR.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 環境の設定
    env_config = {
        "class": CartPoleEnv,
        "wrappers": [
            {"class": LQRMultiBodyEnvWrapper, "init": wrapper["init"]} for wrapper in config["env"]["wrappers"]
        ],
        "init": config["env"]["init"],
        "reset": config["env"]["reset"],
    }

    # エージェントの設定
    agent_config = {
        "class": LQRAgent,
        "init": config["agent"]["init"],
        "reset": config["agent"]["reset"],
    }

    # バッファの設定
    buffer_config = {
        "class": Buffer,
        "init": config["buffer"]["init"],
        "reset": config["buffer"]["reset"],
    }

    runner_config = {
        "class": Runner,
        "init": config["runner"]["init"],
        "reset": config["runner"]["reset"],
        "evaluate": config["runner"]["evaluate"],
    }
    runner_config["evaluate"]["is_render"] = False

    # パレート最適解集合の計算
    Q = np.linspace(0.0, 10.0, 6)[1:]
    R = np.linspace(0.0, 10.0, 6)[1:]
    sse_states = []
    sse_us = []
    sse_final_states = []
    for Q, R in tqdm(zip(*[val.reshape(-1) for val in np.meshgrid(Q, R)]), total=25):
        # Q, Rの設定
        env_config["reset"]["Q"] = Q
        env_config["reset"]["R"] = R
        agent_config["reset"]["Q"] = Q
        agent_config["reset"]["R"] = R

        # Runnerの設定
        runner: Runner = runner_config["class"](env_config, agent_config, buffer_config, **config["runner"]["init"])
        runner.reset(**runner_config["reset"])

        # パレート最適解の計算
        runner.evaluate(**runner_config["evaluate"])
        t_interval = runner.evaluate_result.get()["t"][1] - runner.evaluate_result.get()["t"][0]
        states = np.array(runner.evaluate_result.get()["s"], dtype=np.float64)
        us = np.array(runner.evaluate_result.get()["u"], dtype=np.float64)
        sse_state = np.sum(np.linalg.norm(states[:-1], axis=1) ** 2) * t_interval
        sse_u = np.sum(np.linalg.norm(us, axis=1) ** 2) * t_interval
        sse_final_state = np.linalg.norm(states[-1]) ** 2

        # 結果の追加
        sse_states.append(sse_state)
        sse_us.append(sse_u)
        sse_final_states.append(sse_final_state)

        # Close runner
        runner.close()
    sse_states = np.array(sse_states, dtype=np.float64)
    sse_us = np.array(sse_us, dtype=np.float64)
    sse_final_states = np.array(sse_final_states, dtype=np.float64)

    # データの保存
    savedir = os.path.join(config["runner"]["save"]["savedir"], "pareto_optimal_solutions")
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "sum_square_errors.pickle"), "wb") as f:
        pickle.dump({"s": sse_states, "u": sse_us, "final_s": sse_final_states}, f)
    figure_data = {
        "x": {
            "label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{x}^T\\boldsymbol{x}dt$",
            "value": sse_states,
        },
        "y": {"label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{u}^T\\boldsymbol{u}dt$", "value": sse_us},
        "z": {"label": "$\\boldsymbol{x}^T(t_f)\\boldsymbol{x}(t_f)$", "value": sse_final_states},
    }
    savefile = "pareto_optimal_solutions.png"
    figure_maker = FigureMaker3d()
    figure_maker.reset()
    figure_maker.make(figure_data)
    figure_maker.save(savedir, savefile=savefile)
    figure_maker.close()
