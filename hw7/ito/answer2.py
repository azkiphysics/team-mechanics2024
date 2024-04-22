import os
import yaml

import numpy as np
from tqdm import tqdm

from answer1 import Runner
from common.agents import LQRAgent
from common.buffers import Buffer
from common.envs import CartPoleEnv
from common.utils import FigureMaker
from common.wrappers import LQRMultiBodyEnvWrapper


if __name__ == "__main__":
    config_path = os.path.join("configs", "CartPoleEnv", "LQR.yaml")
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

    # パレート最適解集合の計算
    sse_states = []
    sse_us = []
    for R in tqdm(np.linspace(0.0, 10.0, 101)[1:]):
        # Q, Rの設定
        env_config["reset"]["Q"] = 1.0
        env_config["reset"]["R"] = R
        agent_config["reset"]["Q"] = 1.0
        agent_config["reset"]["R"] = R

        # Runnerの設定
        runner = Runner(env_config, agent_config, buffer_config, **config["runner"]["init"])
        runner.reset(**config["runner"]["reset"])

        # パレート最適解の計算
        runner.evaluate(**config["runner"]["evaluate"])
        states = np.array(runner.evaluate_result.get()["s"], dtype=np.float64)
        us = np.array(runner.evaluate_result.get()["u"], dtype=np.float64)
        sse_state = np.sum(np.linalg.norm(states, axis=1) ** 2) * runner.env.unwrapped.dt
        sse_u = np.sum(np.linalg.norm(us, axis=1) ** 2) * runner.env.unwrapped.dt

        # 結果の追加
        sse_states.append(sse_state)
        sse_us.append(sse_u)

        # Close runner
        runner.close()
    sse_states = np.array(sse_states, dtype=np.float64)
    sse_us = np.array(sse_us, dtype=np.float64)

    # データの保存
    figure_data = {
        "x": {"label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{x}^T\\boldsymbol{x}dt$", "value": sse_states},
        "y": {"label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{u}^T\\boldsymbol{u}dt$", "value": sse_us},
    }
    savedir = config["runner"]["save"]["savedir"]
    savefile = "pareto_optimal_solutions.png"
    figure_maker = FigureMaker()
    figure_maker.reset()
    figure_maker.make(figure_data, draw_type="scatter")
    figure_maker.save(savedir, savefile=savefile)
