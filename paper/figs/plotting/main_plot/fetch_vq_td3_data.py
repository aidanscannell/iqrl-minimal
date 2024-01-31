# %%
ddpg_path = "aalto-ml/lifelong-td3-tc/"
ddpg_data = {
    "path": ddpg_path,
    "data": {
        # 'acrobot-swingup': ['1eiys0vf', '1yedk0zx', '1pbg10k5', 'zgb6ndd1', '2dh8jyh9',],
        # "cheetah-run": ["6bzzn892"],  # "lcmbhw37"],
        "humanoid-walk": ["lcmbhw37"],  # "lcmbhw37"],
        "humanoid-run": ["jv7cfy2v"],
        # 'fish-swim': ['1zrla7u4', '3rx4mc18', 'xwq0fu1r', 'smv9miz4', 'nu50jdp0', ],
        "cartpole-swingup": ["w6eykaow"],  # UTD=1
        "hopper-stand": ["3ozramwe"],  # UTD=1
        "quadruped-walk": ["8q59zkws"],  # UTD=1
        # "cheetah-run": ["mkmvf5md"],  # UTD=4
        "cheetah-run": ["jc3gjwdg"],  # UTD=1
        "walker-walk": ["26lcg142"],  # UTD=1
        # "walker-walk": ["nrak2mn1"],  # UTD=4
        "quadruped-run": ["478dl8sb"],  # UTD=1
        # 'humanoid-walk': ['2xa4zh7i', 'u01r5cd8', '2dm45ezi', '2ywqbphi', '33kix5gc', ],
        # "dog-walk": ["gtrhfxn3"],
        "dog-run": ["ukzsy2t5"],
        "dog-walk": ["sjqd8i9u"],
        "dog-run": ["zhsh5oyd"],
    },
}

# %%
# Plot wandb
import numpy as np
from typing import Dict

import matplotlib.pyplot as plt

# plt.style.use("seaborn")
plt.style.use("seaborn-v0_8")

import wandb

api = wandb.Api(timeout=19)
import pandas as pd

# pet-pytorch
titles = [
    "env_step",
    # "episode",
    "episode_reward",
    # "time",
]
keys = [
    "eval/.env_step",
    # "eval/.episode",
    "eval/.episodic_return",
    # "eval/.eval_total_time",
]

rename = {a: b for a, b in zip(keys, titles)}


def fetch_results(run_path, run_name_list, keys, agent_name):
    data = []
    for run_name in run_name_list:
        wandb_run = api.run(run_path + run_name)
        # history = wandb_run.history(keys=keys, pandas=True)
        history = wandb_run.history(keys=keys)
        history = history.rename(columns=rename)
        # obtain seed, and env_name
        seed = wandb_run.config["seed"]
        env_id = wandb_run.config["env_id"]
        dmc_task = wandb_run.config["dmc_task"]
        env_name = env_id + "-" + dmc_task

        # append env_name, seed, and agent_name
        # history[""] = env_name
        history["episode"] = 0
        history["eval_total_time"] = 0
        history["frame"] = 0
        history["time"] = 0
        history["env"] = env_name
        history["seed"] = seed
        history["agent"] = agent_name

        data.append(history)
        # breakpoint()
    return pd.concat(data)


# %%
data = pd.concat(
    [
        fetch_results(
            run_path=ddpg_data["path"],
            run_name_list=ddpg_data["data"][env],
            keys=keys,
            agent_name="VQ-TD3",
        )
        for env in ddpg_data["data"].keys()
    ]
)

# %%

data.to_csv("./vq_td3_main.csv")


# %%
