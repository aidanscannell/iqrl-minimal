# %%
ddpg_path = "aalto-ml/lifelong-td3-tc/"
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
        "hopper-stand": [
            # no-norm, d=50
            "25m4987b",  # seed=1
            "cet5h22j",  # seed=2
            "8o4v1gz1",  # seed=3
            "udj85ac1",  # seed=4
            "7k9kjfll",  # seed=5
            # no-norm, d=512
            "hnbr2b0o",  # seed=1
            "el8y18hl",  # seed=2
            "1dx4600a",  # seed=3
            "vxw4sgkb",  # seed=4
            "ozh83jpd",  # seed=5
            # norm, d=512
            "1wmalfbs",  # seed=1
            "rj3vhnbq",  # seed=2
            "j4iwbxq2",  # seed=3
            "jo9td3sx",  # seed=4
            "71nsurzh",  # seed=5
        ],
        "quadruped-run": [
            # no-norm, d=50
            "lfay8jdo",  # seed=1
            "r40qcslq",  # seed=2
            "sgc9fa90",  # seed=3
            "fm07n1cn",  # seed=4
            "bbmqgmfm",  # seed=5
            # no-norm, d=512
            "cuaukeiq",  # seed=1
            "n8xdctnq",  # seed=2
            "afuu4r4x",  # seed=3
            "a9qzumnn",  # seed=4
            "drhxy0x4",  # seed=5
            # norm, d=512
            "w625abb9",  # seed=1
            "w1ssoh9i",  # seed=2
            "wtobbh3o",  # seed=3
            "4sj3p04r",  # seed=4
            "wtobbh3o",  # seed=5
        ],
        "walker-run": [
            # no-norm, d=50
            "mkkk0ued",  # seed=1
            "3875xe79",  # seed=2
            "91tzu8ic",  # seed=3
            "35y9q3sl",  # seed=4
            "3bsl2nk7",  # seed=5
            # no-norm, d=512
            "mxm3wbtx",  # seed=1
            "jz8qcbhj",  # seed=2
            "k3fqq3ct",  # seed=3
            "6ryleoei",  # seed=4
            "8ojbiyw2",  # seed=5
            # norm, d=512
            "93c29we8",  # seed=1
            "kxf7upys",  # seed=2
            "v6elb6zj",  # seed=3
            "3rs95mmd",  # seed=4
            "uub5tiz1",  # seed=5
        ],
        "cheetah-run": [
            # no-norm, d=50
            "5qubmpxz",  # seed=1
            "0td7or9m",  # seed=2
            "ahxjg7v2",  # seed=3
            "ygdphy9l",  # seed=4
            "r8f4trip",  # seed=5
            # no-norm, d=512
            "5zx9auk9",  # seed=1
            "vpl86osp",  # seed=2
            "izli85qp",  # seed=3
            "tstkpt03",  # seed=4
            "pynqal0q",  # seed=5
            # norm, d=512
            "nwusiqnt",  # seed=1
            "pxbu4024",  # seed=2
            "ysrel1rl",  # seed=3
            "kmj3y2ww",  # seed=4
            "bc6sqbdl",  # seed=5
        ],
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
        # history["agent"] = agent_name
        if wandb_run.config["agent"]["use_fsq"]:
            history["name"] = "iFSQ-RL"
        else:
            history["name"] = f"no-norm $d={wandb_run.config['agent']['latent_dim']}$"
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

data.to_csv("./ifsq-rl.csv")


# %%
