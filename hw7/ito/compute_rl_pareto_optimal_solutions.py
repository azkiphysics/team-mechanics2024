import multiprocessing as mp
import os
import pickle
from multiprocessing.connection import Connection
from typing import Literal

import numpy as np
import yaml
from tqdm import tqdm

from answer1 import AGENTS, BUFFERS, ENVS, WRAPPERS, Runner
from common.utils import FigureMaker3d


def execute_rl_training(
    worker_id: int, Q: float, R: float, config: dict[str, dict[str, int | float | str | list]]
) -> tuple[int, float, float, float]:
    # Configの設定
    config["env"]["reset"]["Q"] = Q
    config["env"]["reset"]["R"] = R
    config["env"]["reset"]["w_base"] = R * 25.0 if np.abs(R) > 1e-11 else 2.5
    config["runner"]["evaluate"]["is_render"] = False

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

    # Runnerの設定
    runner = Runner(env_config, agent_config, buffer_config, **runner_config["init"])
    runner.reset(**runner_config["reset"])

    # Runnerの実行
    runner.run(**runner_config["run"])
    # runner.save(**runner_config["save"])

    # パレート最適解の計算
    runner.evaluate(**runner_config["evaluate"])
    t_interval = runner.evaluate_result.get()["t"][1] - runner.evaluate_result.get()["t"][0]
    states = np.array(runner.evaluate_result.get()["s"], dtype=np.float64)
    us = np.array(runner.evaluate_result.get()["u"], dtype=np.float64)
    sse_state = np.sum(np.linalg.norm(states[:-1], axis=1) ** 2) * t_interval
    sse_u = np.sum(np.linalg.norm(us, axis=1) ** 2) * t_interval
    sse_final_state = np.linalg.norm(states[-1]) ** 2
    return worker_id, sse_state, sse_u, sse_final_state


def worker(conn: Connection):
    while True:
        args: tuple[
            Literal["training", "finish", "idling"],
            tuple[int, float, float, dict[str, dict[str, int | float | str | list]]] | None,
        ] = conn.recv()
        command = args[0]
        if command == "training":
            training_args = args[1]
            worker_id, sse_state, sse_u, sse_final_state = execute_rl_training(*training_args)
            conn.send((worker_id, sse_state, sse_u, sse_final_state))
        elif command == "finish":
            break
        elif command == "idling":
            conn.send(None)


if __name__ == "__main__":
    n_workers = 5
    rl_algorithm = "TD3"
    config_path = os.path.join("configs", "CartPoleEnv", "Balance", f"{rl_algorithm}.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    Qs = np.linspace(0.1, 0.7, 5)
    Rs = np.linspace(0.0, 0.8, 5)
    Qs, Rs = [val.reshape(-1) for val in np.meshgrid(Qs, Rs)]

    sse_states = np.zeros_like(Qs, dtype=np.float64)
    sse_us = np.zeros_like(Qs, dtype=np.float64)
    sse_final_states = np.zeros_like(Qs, dtype=np.float64)

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
            parent_conn.send(("training", (next_job, float(Qs[next_job]), float(Rs[next_job]), config)))
            next_job += 1
        with tqdm(total=n_jobs, position=1) as pbar:
            while True:
                for parent_conn in parent_conns:
                    if parent_conn.poll():
                        result = parent_conn.recv()
                        if result is not None:
                            worker_id, sse_state, sse_u, sse_final_state = result
                            k_finished_jobs += 1
                            pbar.update(1)
                            sse_states[worker_id] = sse_state
                            sse_us[worker_id] = sse_u
                            sse_final_states[worker_id] = sse_final_state
                            if next_job < n_jobs:
                                parent_conn.send(
                                    ("training", (next_job, float(Qs[next_job]), float(Rs[next_job]), config))
                                )
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
        savedir = os.path.join(
            "results", "CartPoleEnv", "Balance", rl_algorithm, "scratch", "pareto_optimal_solutions"
        )
        os.makedirs(savedir, exist_ok=True)
        with open(os.path.join(savedir, "sum_square_errors.pickle"), "wb") as f:
            pickle.dump({"Q": Qs, "R": Rs, "s": sse_states, "u": sse_us, "final_s": sse_final_states}, f)
        figure_data = {
            "x": {
                "label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{x}^T\\boldsymbol{x}dt$",
                "value": sse_states,
            },
            "y": {"label": "$\\int_{t=0}^{t_{\\mathrm{max}}}\\boldsymbol{u}^T\\boldsymbol{u}dt$", "value": sse_us},
            "z": {"label": "$\\boldsymbol{x}^T(t_f)\\boldsymbol{x}(t_f)$", "value": sse_final_states},
        }
        figure_maker = FigureMaker3d()
        figure_maker.reset()
        figure_maker.make(figure_data)
        figure_maker.save(savedir, savefile="pareto_optimal_solutions.png")
        figure_maker.close()
    except Exception:
        for parent_conn in parent_conns:
            parent_conn.send(("finish", None))
        for p in processes:
            p.join()
