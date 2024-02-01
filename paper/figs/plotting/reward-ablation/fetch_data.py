# %%
ddpg_path = "kallekku/lifelong-td3-tc/"
ddpg_data = {
    "path": ddpg_path,
    "data": {
        "quadruped-run": [
            #iFSQ-RL
#            "9mbrf8o3",
#            "lffrf7xy",
#            "zcloyygj",
#            "4lmtpd04",
#            "bud72vyt",
            "225j3iqa",
            "gsh5g5ie",
            "wkwqz2hp",
            "1fqkia25",
            "bamy18xg",
            # iFSQL-RL+rew
#            "6jwu9xzm",
#            "3thseoe1",
#            "02tn6r73",
#            "gonirtj6",
#            "e3ak3zco",
            "5fd3wano",
            "zhw57oda",
            "27qykyze",
            "ci0ik0ub",
            "0hlxppg4",
            # iFSQ-RL+cos+rew
#            "sovpuj12",
#            "epkk4mw8",
#            "5mpggr4p",
#            "ams8eapm",
#            "6bg43249",
            "l5mr1ykt",
            "3o91tqjt",
            "tcecjfc5",
            "7giv4t1b",
            "chdxnkyl",
            # TCRL-ours
#            "ijm4oaud",
#            "3bm9cclp",
#            "aa3eaznd",
#            "px4fskkk",
#            "m62ducyj",
            "8nnny8uj",
            "7j4amx4h",
            "ngnrnyes",
            "9ddmzk2n",
            "ekef5rwc",
            # TCRL-ours-small
#            "73kkvwmy",
#            "12krzm4k",
#            "mp0d0djz",
#            "bhixjgby",
#            "6x1suc5x",
        ],
        "cheetah-run": [
            #iFSQ-RL
#            "qwf6w273",
#            "0nabrwhq",
#            "rtmvth22",
#            "ykws26yw",
#            "laqc2684",
            "8p0uvq3g",
            "w72leawq",
            "s6drzdek",
            "kwy8sy5b",
            "7w91lnrg",
            # iFSQL-RL+rew
#            "3swdo49x",
#            "4dyxg9qk",
#            "8wddh540",
#            "9w6qirjv",
#            "s6tcdems",
            "7orxmziz",
            "ab4p90vm",
            "xdkltyip",
            "eme4qz1k",
            "52v0lgi8",
            # iFSQ-RL+cos+rew
#            "1cwnzswz",
#            "x5kfrja6",
#            "uvgz27v5",
#            "q0k1lipu",
#            "76wve8ur",
            "acjxczcb",
            "uaqeo9gw",
            "19av8cmt",
            "tvrtbmcy",
            "juqa3xv8",
            # TCRL-ours
#            "vdokw9jq",
#            "pgraxf9o",
#            "liovvbxl",
#            "enf1yxog",
#            "9dvyimy2",
            "r7fyun9l",
            "sh6ruovu",
            "cawuzprz",
            "lrxqkykg",
            "umev07ef",
            # TCRL-ours-small
#            "uf6zvs7x",
#            "rf3byrp3",
#            "9finghq7",
#            "msaw8p2y",
#            "ij15zhei",
        ],
        "walker-run": [
            #iFSQ-RL
#            "3jb600ti",
#            "figec17w",
#            "gjxlbwt2",
#            "tz44nrf2",
#            "ry4x7apa",
            "x36un6ef",
            "1u6zw0u9",
            "14jxzvlp",
            "mbe1a84k",
            "mbdqn9jv",
            # iFSQL-RL+rew
#            "mqvkykyt",
#            "46z9excc",
#            "igkyj5o7",
#            "kholfg79",
#            "parvas63",
            "a4pafxb1",
            "dwyzzbxb",
            "ql43hw7n",
            "ao1aqzit",
            "ennbisac",
            # iFSQ-RL+cos+rew
#            "bx97od2a",
#            "2ow4nou8",
#            "mba294qj",
#            "hdeowx5i",
#            "e5tei16a",
            "1lo0mufa",
            "fn0wy3vr",
            "wg5jth0h",
            "7nhf7hwt",
            "k6943f1f",
            # TCRL-ours
#            "c26cchq4",
#            "6q303gik",
#            "udnvgnn5",
#            "g4oo88c6",
#            "2f2t3thx",
            "09z33fdi",
            "ne25yt6h",
            "imbhze48",
            "8l1gkitp",
            "psl6d2s2",
            # TCRL-ours-small
#            "v8bw5z2s",
#            "pigqs5t6",
#            "pxntj38s",
#            "keujnn21",
#            "0bdgszcr",
        ],
        "fish-swim": [
            #iFSQ-RL
#            "14xdimoh",
#            "qziervk1",
#            "92xdroib",
#            "a31ojzbq",
#            "w6nbzm8s",
            "qan3qdrz",
            "qzatnvv6",
            "1ml0gf0u",
            "s6bm3ob5",
            "fjxj3a9c",
            # iFSQL-RL+rew
#            "p2sktykm",
#            "ehfdb1w6",
#            "n9ux2jna",
#            "woritvoc",
#            "vajvgi9h",
            "43jlszuh",
            "gtxx2gb6",
            "dp4oo35i",
            "rcx5k59s",
            "gyb76ln0",
            # iFSQ-RL+cos+rew
#            "cx5hp70e",
#            "9h6c4w1y",
#            "buzllzsj",
#            "q6bq6kwg",
#            "lv73ouro",
            "x4k2xcqs",
            "mxubqto1",
            "wzagg0f9",
            "qs2nif4x",
            "y7thjq01",
            # TCRL-ours
#            "u9g3yjzt",
#            "kwt2x6qj",
#            "zdr8wz4y",
#            "ijoau9qk",
#            "gj1pyuz6",
            "llpw6ylh",
            "woa3ersg",
            "tappvaxo",
            "1elrkpgk",
            "xe9kntwb",
            # TCRL-ours-small
#            "z27bxa52",
#            "wxsc2hlu",
#            "uklgdoza",
#            "9ixdohkw",
#            "7rxwzh11",
        ],
        "hopper-stand": [
            #iFSQ-RL
#            "fcaei9lc",
#            "9a4u7a5m",
#            "z9wxrjip",
#            "55ukhhd1",
#            "y975o9ny",
            "ndnpces2",
            "ocljrjub",
            "n0djjfyi",
            "b16mtwk1",
            "p9tsyk39",
            # iFSQL-RL+rew
#            "34y4kpae",
#            "ltyurfic",
#            "y0cvhsgv",
#            "5b3initt",
#            "rzn223th",
            "ij5mvgd1",
            "2d39ytqa",
            "qmkzz1lp",
            "vascgcl8",
            "enp61puj",
            # iFSQ-RL+cos+rew
#            "n6vff11u",
#            "hytvufnn",
#            "1rxfjklx",
#            "x8cwznqm",
#            "edwpgjg9",
            "d5uyuyyt",
            "pxuuaywo",
            "ffma7aqi",
            "zlv75x90",
            "fjoc3ug7",
            # TCRL-ours
#            "9yc7a4t5",
#            "8i02940q",
#            "6b952z7p",
#            "164z0w9o",
#            "skgs35h5",
            "iajzh9fx",
            "fitzzesx",
            "rso667hx",
            "nyn67mr1",
            "3jc5valo",
            # TCRL-ours-small
#            "a0lqvtnf",
#            "ew1iuzfn",
#            "kpunfmr4",
#            "xjv1qoj7",
#            "o7570gdq",
       ],
        "acrobot-swingup": [
            #iFSQ-RL
            "51kk2n3c",
            "erw6j34g",
            "10dblk1f",
            "7wj1j3v5",
            "8esvas9g",
            # iFSQL-RL+rew
            "igl1ryke",
            "luhpyp1a",
            "x5gxogow",
            "0hy5gyqv",
            "ttxxx2c2",
            # iFSQ-RL+cos+rew
            "ntzehl2m",
            "nsnjwlah",
            "n8299o2a",
            "jw0ais3m",
            "jlcn2b8j",
            # TCRL-ours
            "08f87wi1",
            "bp8bb3j7",
            "hk2bra8w",
            "hfcgxpyr",
            "g2827ny7",
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
            history["name"] = "iQRL"
        elif not wandb_run.config["agent"]["use_cosine_similarity_dynamics"]:
            history["name"] = "iQRL+rew"
        elif wandb_run.config["agent"]["use_cosine_similarity_dynamics"] and \
                wandb_run.config["agent"]["use_fsq"]:
            history["name"] = "iQRL+cos+rew"
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
