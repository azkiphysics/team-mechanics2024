import multiprocessing as mp
import os
import pickle
from multiprocessing.connection import PipeConnection
from typing import Dict, Literal, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from answer1 import AGENTS, BUFFERS, ENVS, WRAPPERS, Runner
from common.utils import FigureMaker


def execute_rl_training(
    worker_id: int, Q: float, R: float, config: Dict[str, Dict[str, int | float | str | list]]
) -> Tuple[int, float, float]:
    # 環境の設定
    env_config = {
        "class": ENVS[config["env"]["name"]],
        "wrappers": [
            {"class": WRAPPERS[wrapper["name"]], "init": wrapper["init"]} for wrapper in config["env"]["wrappers"]
        ],
        "init": config["env"]["init"],
        "reset": config["env"]["reset"],
    }
    env_config["reset"]["Q"] = Q
    env_config["reset"]["R"] = R
    env_config["reset"]["w_base"] = R * 20.0 if np.abs(R) > 1e-11 else 2.5

    # エージェントの設定
    agent_config = {
        "class": AGENTS[config["agent"]["name"]],
        "init": config["agent"]["init"],
        "reset": config["agent"]["reset"],
    }

    # バッファの設定
    buffer_config = {
        "class": BUFFERS[config["buffer"]["name"]],
        "init": config["buffer"]["init"],
        "reset": config["buffer"]["reset"],
    }

    # Runnerの設定
    runner_config = config["runner"]
    runner_config["evaluate"]["is_render"] = False
    Qf = env_config["reset"]["Qf"]
    runner_config["save"]["savedir"] = os.path.join(
        "results",
        "CartPoleEnv",
        "Balance",
        "TD3",
        "scratch",
        "pareto_optimal_solutions",
        f"Q_{Q: .1f}_R_{R: .1f}_Qf_{Qf: .1f}",
    )

    # Runnerの設定
    runner = Runner(env_config, agent_config, buffer_config, **runner_config["init"])
    runner.reset(**runner_config["reset"])

    # Runnerの実行
    runner.run(**runner_config["run"])
    runner.save(**runner_config["save"])

    # パレート最適解の計算
    runner.evaluate(**runner_config["evaluate"])
    t_interval = runner.evaluate_result.get()["t"][1] - runner.evaluate_result.get()["t"][0]
    states = np.array(runner.evaluate_result.get()["s"], dtype=np.float64)
    us = np.array(runner.evaluate_result.get()["u"], dtype=np.float64)
    sse_state = np.sum(np.linalg.norm(states[:-1], axis=1) ** 2) * t_interval + np.linalg.norm(states[-1]) ** 2
    sse_u = np.sum(np.linalg.norm(us, axis=1) ** 2) * t_interval
    return worker_id, sse_state, sse_u


def worker(conn: PipeConnection):
    while True:
        args: Tuple[
            Literal["training", "finish", "idling"],
            Tuple[int, float, float, Dict[str, Dict[str, int | float | str | list]]] | None,
        ] = conn.recv()
        command = args[0]
        if command == "training":
            training_args = args[1]
            worker_id, sse_state, sse_u = execute_rl_training(*training_args)
            conn.send((worker_id, sse_state, sse_u))
        elif command == "finish":
            break
        elif command == "idling":
            conn.send(None)


if __name__ == "__main__":
    n_workers = 5
    config_path = os.path.join("configs", "CartPoleEnv", "Balance", "TD3.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    Qs = np.linspace(0.2, 1.0, 5)
    Rs = np.linspace(0.0, 1.0, 6)[:-1]
    Qs, Rs = [val.reshape(-1) for val in np.meshgrid(Qs, Rs)]

    sse_states = np.zeros_like(Qs, dtype=np.float64)
    sse_us = np.zeros_like(Qs, dtype=np.float64)

    try:
        ctx = mp.get_context("spawn")
        parent_conns, child_conns = zip(*[ctx.Pipe() for _ in range(n_workers)])
        processes = [ctx.Process(target=worker, args=(child_conn,)) for child_conn in child_conns]
        for p in processes:
            p.start()
        n_jobs = len(Qs)
        next_job = 0
        k_finished_jobs = 0
        for parent_conn in parent_conns:
            parent_conn.send(("training", (next_job, Qs[next_job], Rs[next_job], config)))
            next_job += 1
        with tqdm(total=n_jobs, position=1) as pbar:
            while True:
                for parent_conn in parent_conns:
                    result = parent_conn.recv()
                    if result is not None:
                        worker_id, sse_state, sse_u = result
                        k_finished_jobs += 1
                        pbar.update(1)
                        sse_states[worker_id] = sse_state
                        sse_us[worker_id] = sse_u
                        if next_job < n_jobs:
                            parent_conn.send(("training", (next_job, Qs[next_job], Rs[next_job], config)))
                            next_job += 1
                        else:
                            parent_conn.send(("idling", None))
                    else:
                        parent_conn.send(("idling", None))
                if k_finished_jobs >= n_jobs:
                    break
        for parent_conn in parent_conns:
            parent_conn.send(("finish", None))
        for p in processes:
            p.join()

        # データの保存
        savedir = os.path.join("results", "CartPoleEnv", "Balance", "TD3", "scratch", "pareto_optimal_solutions")
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
    except Exception:
        for parent_conn in parent_conns:
            parent_conn.send(("finish", None))
        for p in processes:
            p.join()
