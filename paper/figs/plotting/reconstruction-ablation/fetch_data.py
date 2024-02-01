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
            ##### Project=True
            # # reconstruction
            # "25m4987b",  # seed=1
            # "cet5h22j",  # seed=2
            # "8o4v1gz1",  # seed=3
            # "udj85ac1",  # seed=4
            # "7k9kjfll",  # seed=5
            # # no reconstruction
            # "1wmalfbs",  # seed=1
            # "rj3vhnbq",  # seed=2
            # "j4iwbxq2",  # seed=3
            # "jo9td3sx",  # seed=4
            # "71nsurzh",  # seed=5
            # reconstruction
            "94skam71",  # seed=1
            "hss7p9jr",  # seed=2
            "oe1us2qh",  # seed=3
            "nijlewdk",  # seed=4
            "4m35m7d4",  # seed=5
            # no reconstruction
            "3rmiy5he",  # seed=1
            "39j1gbp2",  # seed=2
            "t4hf7g18",  # seed=3
            "j8apohl8",  # seed=4
            "5w8halqc",  # seed=5
        ],
        "quadruped-run": [
            ##### Project=True
            # # reconstruction
            # "rw26pb1l",  # seed=1
            # "jm549p6m",  # seed=2
            # "7ll3mp6d",  # seed=3
            # "ypgoji3k",  # seed=4
            # "ll54kaha",  # seed=5
            # # no reconstruction
            # "w625abb9",  # seed=1
            # "w1ssoh9i",  # seed=2
            # "wtobbh3o",  # seed=3
            # "4sj3p04r",  # seed=4
            # "wtobbh3o",  # seed=5
            ##### Project=False
            # reconstruction
            "3yyb8ibb",  # seed=1
            "v0e4moj8",  # seed=2
            "r9ld9h3r",  # seed=3
            "sats7q1b",  # seed=4
            "foax0pto",  # seed=5
            # no reconstruction
            "g16tux3h",  # seed=1
            "82tqwdmq",  # seed=2
            "msec1uul",  # seed=3
            "jd6nsd17",  # seed=4
            "i09us6qm",  # seed=5
        ],
        "walker-run": [
            ##### Project=True
            # # reconstruction
            # "qh3xxjz0",  # seed=1
            # "1hjix0d3",  # seed=2
            # "7agg1jz9",  # seed=3
            # "dnqzhj7e",  # seed=4
            # "kabd20ca",  # seed=5
            # # no reconstruction
            # "93c29we8",  # seed=1
            # "kxf7upys",  # seed=2
            # "v6elb6zj",  # seed=3
            # "3rs95mmd",  # seed=4
            # "uub5tiz1",  # seed=5
            ##### Project=False
            # reconstruction
            "1xghgx8k",  # seed=1
            "u9530uao",  # seed=2
            "def6c7jo",  # seed=3
            "pv28an1a",  # seed=4
            "rswp0leu",  # seed=5
            # no reconstruction
            "npwa34kk",  # seed=1
            "8vt2l9wn",  # seed=2
            "dnhbpv2d",  # seed=3
            "w65qql3j",  # seed=4
            "wkge5rid",  # seed=5
        ],
        "cheetah-run": [
            ##### Project=True
            # # reconstruction
            # "bgimfets",  # seed=1
            # "juw5xc9u",  # seed=2
            # "0klyyvx0",  # seed=3
            # "n44sclk6",  # seed=4
            # "u5j8xtm1",  # seed=5
            # # no reconstruction
            # "nwusiqnt",  # seed=1
            # "pxbu4024",  # seed=2
            # "ysrel1rl",  # seed=3
            # "kmj3y2ww",  # seed=4
            # "bc6sqbdl",  # seed=5
            # reconstruction
            "0jds01bs",  # seed=1
            "8dqg9vxh",  # seed=2
            "fubrnbti",  # seed=3
            "q7ib7ovd",  # seed=4
            "ndtfn06e",  # seed=5
            # no reconstruction
            ##### Project=False
            "7kg8xyj8",  # seed=1
            "gujw41jb",  # seed=2
            "r9cu6yvh",  # seed=3
            "lh1kdlht",  # seed=4
            "8gymqblt",  # seed=5
        ],
        "dog-walk": [
            # # reconstruction (project_latent=True)
            # "lvpwhw2a",  # seed=1
            # "grfre0ts",  # seed=2
            # "vk2roup1",  # seed=3
            # "r02k6u8t",  # seed=4
            # "8hks5538",  # seed=5
            # reconstruction (project_latent=False)
            "9h63x0wt",  # seed=1
            "mypst8od",  # seed=2
            "cszonouz",  # seed=3
            "sh2n9br7",  # seed=4
            "wrggk81m",  # seed=5
            # no reconstruction
            # project=False
            "stawiad8",
            "iv8p110s",
            "k454j543",
            "nihkvll9",
            "jpkbw5h7",
            # project=True
            # "9k5cqnuq",
            # "q28f7f9i",
            # "2vf4clxu",
            # "8nnpgifp",
            # "b5ffm8rt",
        ],
        # "dog-walk": ["sjqd8i9u"],
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
        if wandb_run.config["agent"]["reconstruction_loss"]:
            history["name"] = "iFSQ-RL+rec"
        else:
            history["name"] = "iFSQ-RL"

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
