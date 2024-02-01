# %%
ddpg_path = "yizhao/sprl/"
ddpg_data = {
    "path": ddpg_path,
    "data": {
        "acrobot-swingup": [
            "1eiys0vf",
            "1yedk0zx",
            "1pbg10k5",
            "zgb6ndd1",
            "2dh8jyh9",
        ],
        "cheetah-run": [
            "i6sm1a7b",
            "dqax4n78",
            "1p4lrio9",
            "1gfzkhtr",
            "2t0dn97k",
        ],
        "fish-swim": [
            "1zrla7u4",
            "3rx4mc18",
            "xwq0fu1r",
            "smv9miz4",
            "nu50jdp0",
        ],
        "quadruped-walk": [
            "du473356",
            "3kyhmy9n",
            "1yusrrco",
            "3198sjan",
            "3fdn4ijo",
        ],
        "walker-walk": [
            "38b1tu3q",
            "3qg5ghou",
            "2a8wgbv4",
            "kjw1t2hi",
            "c6m4ztsu",
        ],
        "humanoid-walk": [
            "2xa4zh7i",
            "u01r5cd8",
            "2dm45ezi",
            "2ywqbphi",
            "33kix5gc",
        ],
        "dog-walk": [
            "1j5y9f66",
            "10k9mjuf",
            "er1d5h4k",
            "t6b44j79",
            "3500wzl6",
        ],
        "dog-run": [
            "2e43rg7f",
            "1ev56klj",
            "1rtay2kk",
            "1fjb7wss",
            "2p7hov2d",
        ],
    },
}

# %%
# Plot wandb
import numpy as np
from typing import Dict

import matplotlib.pyplot as plt

# plt.style.use('seaborn')
plt.style.use("seaborn-v0_8")

import wandb

api = wandb.Api(timeout=19)
import pandas as pd

# pet-pytorch
titles = [
    "env_step",
    "episode",
    "episode_reward",
    "time",
]
keys = [
    "eval/.env_step",
    "eval/.episode",
    "eval/.episode_reward",
    "eval/.eval_total_time",
]

rename = {a: b for a, b in zip(keys, titles)}


def fetch_results(run_path, run_name_list, keys, agent_name):
    data = []
    for run_name in run_name_list:
        wandb_run = api.run(run_path + run_name)
        history = wandb_run.history(keys=keys, pandas=True)
        history = history.rename(columns=rename)
        # obtain seed, and env_name
        seed, env_name = (
            wandb_run.config["_content"]["seed"],
            wandb_run.config["_content"]["env_name"],
        )

        # append env_name, seed, and agent_name
        history["env"] = env_name
        history["seed"] = seed
        history["agent"] = agent_name

        data.append(history)
    return pd.concat(data)


# %%
data = pd.concat(
    [
        fetch_results(
            run_path=ddpg_data["path"],
            run_name_list=ddpg_data["data"][env],
            keys=keys,
            agent_name="TD3",
        )
        for env in ddpg_data["data"].keys()
    ]
)

# %%

data.to_csv("./td3_main.csv")


# %%
