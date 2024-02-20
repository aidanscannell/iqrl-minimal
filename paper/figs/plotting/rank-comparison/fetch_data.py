# %%
ddpg_path = "aalto-ml/iqrl-icml/"
ddpg_data = {
    "path": ddpg_path,
    "data": {
        # 'acrobot-swingup': ['1eiys0vf', '1yedk0zx', '1pbg10k5', 'zgb6ndd1', '2dh8jyh9',],
        # "cheetah-run": ["6bzzn892"],  # "lcmbhw37"],
        # "humanoid-walk": ["lcmbhw37"],  # "lcmbhw37"],
        # "humanoid-run": ["jv7cfy2v"],
        # 'fish-swim': ['1zrla7u4', '3rx4mc18', 'xwq0fu1r', 'smv9miz4', 'nu50jdp0', ],
        # "cartpole-swingup": ["w6eykaow"],  # UTD=1
        # "hopper-stand": ["3ozramwe"],  # UTD=1
        # "quadruped-walk": ["8q59zkws"],  # UTD=1
        # "cheetah-run": ["mkmvf5md"],  # UTD=4
        # : [,],
        "cheetah-run-d128-no-norm": [
            "5j2it0tc",
            "v0fz1014",
            "v1i6mftd",
            "yfkyy6rt",
            "txtebz1w",
        ],
        "cheetah-run-d128-norm": [
            "l4c3phty",
            "96xm0zx5",
            "6tul0lmk",
            "y7c9k82p",
            "zx6ykgj9",
        ],
        "hopper-stand-d=64-no-norm": [
            "0ymb3imw",
            "5fibx0fg",
            "i8ou5zx4",
            "bijgsw6i",
            "02uvlbjb",
        ],
        "hopper-stand-d=64": [
            "b978u8ed",
            "77bswwl0",
            "rkdmf32p",
            "ucvgujrp",
            "zqv19l3p",
        ],
        "humanoid-run": [
            # no-norm d=1024 project=False
            "uuyjutba",
            "f1wdhmr1",
            "cbrtltx4",
            "arilzq7a",
            "ua5c03do",
            # norm, d=1024 projection=False
            "dt0r1wyb",
            "g35s9yyg",
            "fuqpil5h",
            "15i70x9w",
            "agjdi84a",
        ],
        "acrobot-swingup": [
            # no norm
            "vkbkcrjm",
            "nx50by5c",
            "4iy9ahwx",
            "frc6lm7a",
            "e730qvy7",
            # norm
            "c6qmc8yx",
            "spwng8k4",
            "wgurvdan",
            "4xfe6bez",
            "95l445ci",
        ],
        "fish-swim-no-norm": [
            "a2r96nmy",
            "1cezxqig",
            "15r32rsc",
            "rxyyhje0",
            "8pho3asn",
        ],
        "fish-swim": ["3duujm1p", "kl6xd92o", "s4t5viko", "flcmnkt7", "1zjl534e"],
        # 'humanoid-walk': ['2xa4zh7i', 'u01r5cd8', '2dm45ezi', '2ywqbphi', '33kix5gc', ],
        # "dog-walk": ["gtrhfxn3"],
        # "dog-run": ["ukzsy2t5"],
        # "dog-walk": ["sjqd8i9u"],
        # "dog-run": ["zhsh5oyd"],
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
    "rank0",
    "rank1",
    "rank2",
]
keys = [
    "eval/.env_step",
    "eval/.episode",
    "eval/.episodic_return",
    "eval/.elapsed_time",
    "eval/.z-rank-0",
    "eval/.z-rank-1",
    "eval/.z-rank-2",
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
        # history["agent"] = agent_name
        if wandb_run.config["agent"]["use_fsq"]:
            history["name"] = "iQRL"
        else:
            history["name"] = "iQRL-no-normalization"
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
