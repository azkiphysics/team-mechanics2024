import pickle

import numpy as np

from common.utils import FigureMaker

savedir = "results\\CartPoleEnv\\TD3\\run"
path = savedir + "\\training_data.pickle"
with open(path, "rb") as f:
    data = pickle.load(f)


print(data.keys())
episodes = np.array(data["episode"])
total_rewards = np.array(data["total_rewards"])

new_episodes = 1 + episodes[:-1][episodes[1:] > episodes[:-1]]
new_total_rewards = total_rewards[:-1][episodes[1:] > episodes[:-1]]
total_rewards_data = {
    "x": {"label": "Episode", "value": new_episodes},
    "y": {
        "label": "Total rewards",
        "value": new_total_rewards,
    },
}

figure_maker = FigureMaker()
figure_maker.reset()
figure_maker.make(total_rewards_data)
figure_maker.save("results\\CartPoleEnv\\TD3\\run", "total_rewards.png")

data["episode"] = list(new_episodes)
data["total_rewards"] = list(new_total_rewards)

print(data["episode"][-1], len(data["episode"]), len(data["total_rewards"]))

path = savedir + "\\new_training_data.pickle"
with open(path, "wb") as f:
    pickle.dump(data, f)
