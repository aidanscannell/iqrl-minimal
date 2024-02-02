# %%
ddpg_path = "kallekku/lifelong-td3-tc/"
ddpg_data = {
    "path": ddpg_path,
    "data": {
        "humanoid-run": [
            # Pre-training
            "h0e96eeo",
            "6tykwpqn",
            "w446n5he",
            "fnzw5jyj",
            "ugcwkueb",
            # No pre-training:
            "ek6t7mtb",
            "febl1741",
            "i2926y8u",
            "jntea203",
            "kjw08gye",
        ],
        "hopper-hop": [
            # Pre-training
            "dblmpj4q",
            "hx4frieq",
            "j13wyns5",
            "eonukhi9",
            "s9br2god",
            # No pre-training
            "4ax1gct6",
            "f5d66x17",
            "jp26qyvm",
            "n3cb46el",
            "z4dideh5",
        ],
        "quadruped-run": [
            # Pre-training"
            "0zviwszy",
            "dr5wqqs6",
            "0tb5a3p1",
            "jchvk2kd",
            "2l4wphqg",
            # No pre-training
            "1ottm721",
            "67objwkg",
            "a1rml1p2",
            "my89a2ui",
            "0bdbllyg",
            # Task-specific pre-training
            #            "uykvengt",
            #            "e6u4u0et",
            #            "phlr610t",
            #            "32c5rcc6",
            #            "2hyd7evc",
            "lc8kap78",
            "fs0fug53",
            "rypu71yu",
            "7lerwfz0",
            "jcvs7slh",
        ],
        "walker-run": [
            # Pre-training"
            "cua4ibb4",
            "ya38xqbp",
            "1cp2xqb6",
            "vokt0o8m",
            "qwy97zej",
            # No pre-training
            "0y5whgm0",
            "5hhavbpu",
            "zwxgu2fr",
            "cbp72vz9",
            "oi0gnnd1",
            # Task-specific pre-training
            #            "oxisl0xv",
            #            "wv0a1i6l",
            #            "2mstpcgd",
            #            "olnugqts",
            #            "1iegxig2",
            "g029b1dc",
            "2dy32443",
            "1ogr1lye",
            "8rtrpoht",
            "glmvfvho",
        ],
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
    "episode",
    "episode_reward",
    "time",
]
keys = [
    "eval/.env_step",
    "eval/.episode",
    "eval/.episodic_return",
    "eval/.elapsed_time",
    # "use_fsq",
    # "agent.latent_dim",
]

rename = {a: b for a, b in zip(keys, titles)}


def fetch_results(run_path, run_name_list, keys):
    data = []
    for run_name in run_name_list:
        wandb_run = api.run(run_path + run_name)
        # history = wandb_run.history(keys=keys, pandas=True)
        # breakpoint()
        history = wandb_run.history(keys=keys)
        history = history.rename(columns=rename)
        # obtain seed, and env_name
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
        history["seed"] = wandb_run.config["seed"]

        if env_name == "hopper-hop":
            pass
        elif env_name == "humanoid-run":
            history = history[history["env_step"] <= 2e6]
        elif env_name in ["walker-run", "quadruped-run"]:
            history = history[history["env_step"] <= 500000]
        else:
            raise NotImplementedError

        # history["agent"] = agent_name
        if not wandb_run.config["load_pretrained_agent"]:
            history["name"] = "iQRL"
        elif wandb_run.config["agent"]["reward_loss"]:
            history["name"] = "iQRL+rew-pretrained"
        else:
            history["name"] = "iQRL-pretrained"
            # history["name"] = f"no-norm $d={wandb_run.config['agent']['latent_dim']}$"
        history["utd_ratio"] = wandb_run.config["agent"]["utd_ratio"]

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
            # agent_name="VQ-TD3",
        )
        for env in ddpg_data["data"].keys()
    ]
)

# %%

data.to_csv("./iqrl.csv")


# %%
