# %%
ddpg_path = "aalto-ml/iqrl-icml/"
ddpg_data = {
    "path": ddpg_path,
    "data": {
        #### Project=True ####
        # "acrobot-swingup": ["lkjyy7t0", "tek7m4yi", "7vqw7n3a", "4v4cxbjf", "omf199u5"],
        # "cheetah-run": ["nwusiqnt", "pxbu4024", "ysrel1rl", "kmj3y2ww", "bc6sqbdl"],
        "fish-swim": ["5rhervdh", "btrclty9", "royfiwdl", "2bvv2o87", "krfia2hj"],
        # "walker-walk": ["4wydf2sz", "n7l8xac6", "l7nqlsid", "yrwwmyq2", "bpdxi4jw"],
        # "quadruped-walk": ["6gf2r90u", "uvq2ctjr", "4x1r1nqg", "lji8f92s", "eiafkell"],
        #### Project=False ####
        # GOOD performance d=128
        "quadruped-walk": ["8zgkcuhs", "egk6qxco", "oh1xhyqt", "hm3ihqzo", "p4slb4q9"],
        "cheetah-run": ["8x29ibzc", "9k7936z8", "e7afvv6h", "81pbgf97", "qfqv9wz3"],
        # "walker-walk": ["8z9o8vp6", "roz89u0n", "h1by1w6s", "zfu00w22", "jvw4myex"], # only ran for 500k
        "walker-walk": [
            "zgtw2w02",
            "hix7b6di",
            "o17mz4v1",
            "lqvypdgw",
            "ilb76d20",
        ],  # 1M steps
        # "cheetah-run": ["l4c3phty", "96xm0zx5", "6tul0lmk", "y7c9k82p", "zx6ykgj9"],
        # POOR performance d=512 project=False
        "acrobot-swingup": ["c6qmc8yx", "spwng8k4", "wgurvdan", "4xfe6bez", "95l445ci"],
        # "cheetah-run": ["7kg8xyj8", "gujw41jb", "r9cu6yvh", "lh1kdlht", "8gymqblt"],
        # "fish-swim": ["izb29vhj", "xlgm5s8v", "xc1nf2ss", "ezhafy1m", "5tw9ini4"],
        # "walker-walk": ["9xh1l0n1", "9ujhx4vn", "pojcknc3", "2xehepzn", "fvzzgcw2"],
        # "quadruped-walk": ["17h2zwdq", "s6i2xtt3", "fu8wchfa", "tzwhqglg", "7n7tivxv"],
        "humanoid-walk": [
            "wur1u276",
            "l8da5827",
            "oy053sve",
            "krctc38z",
            "anstkvup",
        ],  # project=False
        # "humanoid-walk": [
        #     "bk32bbsi",
        #     "fr1hpr9b",
        #     "t9hbnhff",
        #     "681rgba7",
        #     "ndd0tr73",
        # ],  # project=True
        # "humanoid-walk": ["lcmbhw37"],  # "lcmbhw37"],
        "dog-walk": [
            # "stawiad8",
            "iv8p110s",
            # "k454j543",
            # "nihkvll9",
            "jpkbw5h7",
        ],  # project=False
        # "dog-walk": [
        #     "9k5cqnuq",
        #     "q28f7f9i",
        #     "2vf4clxu",
        #     "8nnpgifp",
        #     "b5ffm8rt",
        # ],  # project=True
        # "dog-walk": ["sjqd8i9u"],
        # "dog-run": [
        #     "ohxmhffe",
        #     "aikn5eix",
        #     "kprj5q9u",
        #     "r232jms2",
        #     # "aox6spn2", # crashed
        #     "5ubkon5e",
        #     "65wmol3k",
        # ],  # project=False
        # "dog-run": [
        #     "7hty9mrz",
        #     "0no65695",
        #     "nfuag4aj",
        #     "otirlyw0",
        #     "t9oyygr9",
        # ],  # project=True
        # "dog-run": ["zhsh5oyd"],
        #### OLD ####
        # "cheetah-run": ["6bzzn892"],  # "lcmbhw37"],
        # "humanoid-walk": ["lcmbhw37"],  # "lcmbhw37"],
        # "humanoid-run": ["jv7cfy2v"],
        # 'fish-swim': ['1zrla7u4', '3rx4mc18', 'xwq0fu1r', 'smv9miz4', 'nu50jdp0', ],
        # "cartpole-swingup": ["w6eykaow"],  # UTD=1
        # "hopper-stand": ["3ozramwe"],  # UTD=1
        # "quadruped-walk": ["8q59zkws"],  # UTD=1
        # "cheetah-run": ["mkmvf5md"],  # UTD=4
        # "cheetah-run": ["jc3gjwdg"],  # UTD=1
        # "walker-walk": ["26lcg142"],  # UTD=1
        # "walker-walk": ["nrak2mn1"],  # UTD=4
        # "quadruped-run": ["478dl8sb"],  # UTD=1
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
        # history["agent"] = agent_name + f" d={wandb_run.config['agent']['latent_dim']}"

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
            agent_name="iQRL",
        )
        for env in ddpg_data["data"].keys()
    ]
)

# %%

data.to_csv("./iqrl.csv")


# %%
