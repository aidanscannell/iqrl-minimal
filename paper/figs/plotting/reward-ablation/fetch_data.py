# %%
ddpg_path = "kallekku/lifelong-td3-tc/"
ddpg_data = {
    "path": ddpg_path,
    "data": {
        "quadruped-run": [
            #iFSQ-RL
            "9mbrf8o3",
            "lffrf7xy",
            "zcloyygj",
            "4lmtpd04",
            "bud72vyt",
            # iFSQL-RL+rew
            "6jwu9xzm",
            "3thseoe1",
            "02tn6r73",
            "gonirtj6",
            "e3ak3zco",
            # iFSQ-RL+cos+rew
            "sovpuj12",
            "epkk4mw8",
            "5mpggr4p",
            "ams8eapm",
            "6bg43249",
            # TCRL-ours
            "ijm4oaud",
            "3bm9cclp",
            "aa3eaznd",
            "px4fskkk",
            "m62ducyj",
            # TCRL-ours-small
            "73kkvwmy",
            "12krzm4k",
            "mp0d0djz",
            "bhixjgby",
            "6x1suc5x",
        ],
        "cheetah-run": [
            #iFSQ-RL
            "qwf6w273",
            "0nabrwhq",
            "rtmvth22",
            "ykws26yw",
            "laqc2684",
            # iFSQL-RL+rew
            "3swdo49x",
            "4dyxg9qk",
            "8wddh540",
            "9w6qirjv",
            "s6tcdems",
            # iFSQ-RL+cos+rew
            "1cwnzswz",
            "x5kfrja6",
            "uvgz27v5",
            "q0k1lipu",
            "76wve8ur",
            # TCRL-ours
            "vdokw9jq",
            "pgraxf9o",
            "liovvbxl",
            "enf1yxog",
            "9dvyimy2",
            # TCRL-ours-small
            "uf6zvs7x",
            "rf3byrp3",
            "9finghq7",
            "msaw8p2y",
            "ij15zhei",
        ],
        "walker-run": [
            #iFSQ-RL
            "3jb600ti",
            "figec17w",
            "gjxlbwt2",
            "tz44nrf2",
            "ry4x7apa",
            # iFSQL-RL+rew
            "mqvkykyt",
            "46z9excc",
            "igkyj5o7",
            "kholfg79",
            "parvas63",
            # iFSQ-RL+cos+rew
            "bx97od2a",
            "2ow4nou8",
            "mba294qj",
            "hdeowx5i",
            "e5tei16a",
            # TCRL-ours
            "c26cchq4",
            "6q303gik",
            "udnvgnn5",
            "g4oo88c6",
            "2f2t3thx",
            # TCRL-ours-small
            "v8bw5z2s",
            "pigqs5t6",
            "pxntj38s",
            "keujnn21",
            "0bdgszcr",
        ],
        "fish-swim": [
            #iFSQ-RL
            "14xdimoh",
            "qziervk1",
            "92xdroib",
            "a31ojzbq",
            "w6nbzm8s",
            # iFSQL-RL+rew
            "p2sktykm",
            "ehfdb1w6",
            "n9ux2jna",
            "woritvoc",
            "vajvgi9h",
            # iFSQ-RL+cos+rew
            "cx5hp70e",
            "9h6c4w1y",
            "buzllzsj",
            "q6bq6kwg",
            "lv73ouro",
            # TCRL-ours
            "u9g3yjzt",
            "kwt2x6qj",
            "zdr8wz4y",
            "ijoau9qk",
            "gj1pyuz6",
            # TCRL-ours-small
            "z27bxa52",
            "wxsc2hlu",
            "uklgdoza",
            "9ixdohkw",
            "7rxwzh11",
        ],
        "hopper-stand": [
            #iFSQ-RL
            "fcaei9lc",
            "9a4u7a5m",
            "z9wxrjip",
            "55ukhhd1",
            "y975o9ny",
            # iFSQL-RL+rew
            "34y4kpae",
            "ltyurfic",
            "y0cvhsgv",
            "5b3initt",
            "rzn223th",
            # iFSQ-RL+cos+rew
            "n6vff11u",
            "hytvufnn",
            "1rxfjklx",
            "x8cwznqm",
            "edwpgjg9",
            # TCRL-ours
            "9yc7a4t5",
            "8i02940q",
            "6b952z7p",
            "164z0w9o",
            "skgs35h5",
            # TCRL-ours-small
            "a0lqvtnf",
            "ew1iuzfn",
            "kpunfmr4",
            "xjv1qoj7",
            "o7570gdq",
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


def fetch_results(run_path, run_name_list, keys, agent_name=None):
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
        if not wandb_run.config["agent"]["reward_loss"]:
            history["name"] = "iFSQ-RL"
        elif not wandb_run.config["agent"]["use_cosine_similarity_dynamics"]:
            history["name"] = "iFSQ-RL-rew"
        elif wandb_run.config["agent"]["use_cosine_similarity_dynamics"] and \
                wandb_run.config["agent"]["use_fsq"]:
            history["name"] = "iFSQ-RL-cos-rew"
        elif (not wandb_run.config["agent"]["use_fsq"]) and \
                wandb_run.config["agent"]["latent_dim"] > 50:
            history["name"] = "TCRL-ours"
        elif (not wandb_run.config["agent"]["use_fsq"]) and \
                wandb_run.config["agent"]["latent_dim"] <= 50:
            history["name"] = "TCRL-ours-small"
        else:
            raise RuntimeError
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
#            agent_name="iFSQ-RL",
        )
        for env in ddpg_data["data"].keys()
    ]
)

# %%

data.to_csv("./ifsq_rl.csv")


# %%
