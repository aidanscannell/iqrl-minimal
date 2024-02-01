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
        "cheetah-run": [
            # UTD=1
            "u9plbcw6",  # seed=64238
            "qh0aoiu8",  # seed=234
            "nsbheioo",  # seed=64823
            "ayf1ov5h",  # seed=7492
            "72d02rea",  # seed=423
            # UTD=2
            "xm9mx9vw",  # seed=64238
            "fy1d9lzj",  # seed=234
            "09g92cv8",  # seed=64823
            "9r14qhgy",  # seed=7492
            "aalwd7ap",  # seed=423
            # UTD=4
            "te46dylh",  # seed=64238
            "rx1rg4fw",  # seed=234
            "p94l3jen",  # seed=64823
            "c5tmc619",  # seed=7492
            "vgawl7me",  # seed=423
            # UTD=8
            "b5cawxro",  # seed=64238
            "cbhkp2og",  # seed=234
            "w7dqbbdz",  # seed=64823
            "pgr6kmbv",  # seed=7492
            "gcx4iet5",  # seed=423
        ],
        "walker-walk": [  # UTD=1
            "ygmyl5ok",  # seed=64238
            "mk1dau1k",  # seed=234
            "iybejvai",  # seed=64823
            "2ncb40mo",  # seed=7492
            "trqix916",  # seed=423
            # UTD=2
            "o9p2rtps",  # seed=64238
            "8i7a9ad1",  # seed=234
            "lpeq7cdp",  # seed=64823
            "vopy7p4w",  # seed=7492
            "sfspo5r0",  # seed=423
            # UTD=4
            "sly5v1io",  # seed=64238
            "cnbwihjl",  # seed=234
            "mdegcl2w",  # seed=64823
            "8op58bu5",  # seed=7492
            "yo4cam1y",  # seed=423
            # UTD=8
            "tne915fs",  # seed=64238
            "zfkt2hn2",  # seed=234
            "8nebkof5",  # seed=64823
            "wuoy51f0",  # seed=7492
            "gxpkha6w",  # seed=423
        ],
        "walker-run": [  # UTD=1
            "py1hagav",  # seed=64238
            "twm4073u",  # seed=234
            "ete1dcb6",  # seed=64823
            "10509jdw",  # seed=7492
            "dioo7bdi",  # seed=423
            # UTD=2
            "lykl5w25",  # seed=64238
            "bdascr9m",  # seed=234
            "8o256394",  # seed=64823
            "e5wpzgu3",  # seed=7492
            "0izief4f",  # seed=423
            # UTD=4
            "syciggcr",  # seed=64238
            "3igwnhw8",  # seed=234
            "h7nfa1gv",  # seed=64823
            "axjcmu2p",  # seed=7492
            "6k7mohsx",  # seed=423
            # UTD=8
            "0vocvhf0",  # seed=64238
            "0v11jgfx",  # seed=234
            "f1jnf0i0",  # seed=64823
            "o281vmu4",  # seed=7492
            "4hpsx8fy",  # seed=423
        ],
        # "humanoid-walk": {
        #     [  # UTD=1
        #         "",  # seed=64238
        #         "",  # seed=234
        #         "",  # seed=64823
        #         "",  # seed=7492
        #         "",  # seed=423
        #     ],
        #     [  # UTD=2
        #         "",  # seed=64238
        #         "",  # seed=234
        #         "",  # seed=64823
        #         "",  # seed=7492
        #         "",  # seed=423
        #     ],
        #     [  # UTD=4
        #         "",  # seed=64238
        #         "",  # seed=234
        #         "",  # seed=64823
        #         "",  # seed=7492
        #         "",  # seed=423
        #     ],
        #     [  # UTD=8
        #         "",  # seed=64238
        #         "",  # seed=234
        #         "",  # seed=64823
        #         "",  # seed=7492
        #         "",  # seed=423
        #     ],
        # },
        "quadruped-run": ["478dl8sb"],  # UTD=1
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
        breakpoint()
        history["agent"] = agent_name
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
            agent_name="VQ-TD3",
        )
        for env in ddpg_data["data"].keys()
    ]
)

# %%

data.to_csv("./vq_td3_main.csv")


# %%
