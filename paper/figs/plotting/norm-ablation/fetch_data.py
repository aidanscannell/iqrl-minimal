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
        "quadruped-run-d=128-no-norm": [
            "q7ia4kx5",
            "i9b9nl8y",
            "tvondn1a",
            "bfbhcgre",
            "kqvznpvy",
        ],
        "quadruped-run-d=128-norm": [
            "csjhne6n",
            "bm5kosik",
            "kum2pe4f",
            "s9p9y44s",
            "4gsq88ju",
        ],
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
        "walker-run-d=128-no-norm": [
            "tkja6y1j",
            "ogmwwcwv",
            "i27uobpy",
            "p59ieaw9",
            "c5a2wliy",
        ],
        "walker-run-d=128": [
            "kb1gk7hm",
            "egjvexs4",
            "lwnz0cf1",
            "zfsg29ej",
            "5k9okav3",
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
        "fish-swim-no-norm": [
            "a2r96nmy",
            "1cezxqig",
            "15r32rsc",
            "rxyyhje0",
            "8pho3asn",
        ],
        "fish-swim": ["3duujm1p", "kl6xd92o", "s4t5viko", "flcmnkt7", "1zjl534e"],
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
        "humanoid-walk": [
            # no-norm d=1024 project=False
            "ccq1197j",
            "q0z8dpq7",
            "uydw5jsw",
            "gjz8tmso",
            "yjt1rd4k",
            # norm, d=1024 projection=False
            "wur1u276",
            "l8da5827",
            "oy053sve",
            "krctc38z",
            "anstkvup",
        ],  # project=False
        # "hopper-stand": [
        #     ###### Project=True
        #     # # no-norm, d=50
        #     # "25m4987b",  # seed=1
        #     # "cet5h22j",  # seed=2
        #     # "8o4v1gz1",  # seed=3
        #     # "udj85ac1",  # seed=4
        #     # "7k9kjfll",  # seed=5
        #     # # no-norm, d=512
        #     # "hnbr2b0o",  # seed=1
        #     # "el8y18hl",  # seed=2
        #     # "1dx4600a",  # seed=3
        #     # "vxw4sgkb",  # seed=4
        #     # "ozh83jpd",  # seed=5
        #     # # norm, d=512
        #     # "1wmalfbs",  # seed=1
        #     # "rj3vhnbq",  # seed=2
        #     # "j4iwbxq2",  # seed=3
        #     # "jo9td3sx",  # seed=4
        #     # "71nsurzh",  # seed=5
        #     ###### Project=False
        #     # no-norm, d=512
        #     "1mv4zcn5",  # seed=1
        #     "5gi6cb5f",  # seed=2
        #     "c8oczcfm",  # seed=3
        #     "wzqwgz8d",  # seed=4
        #     "45qursum",  # seed=5
        #     # norm, d=512
        #     "3rmiy5he",  # seed=1
        #     "39j1gbp2",  # seed=2
        #     "t4hf7g18",  # seed=3
        #     "j8apohl8",  # seed=4
        #     "5w8halqc",  # seed=5
        # ],
        # "quadruped-run": [
        #     ###### Project=True
        #     # # no-norm, d=50
        #     # "lfay8jdo",  # seed=1
        #     # "r40qcslq",  # seed=2
        #     # "sgc9fa90",  # seed=3
        #     # "fm07n1cn",  # seed=4
        #     # "bbmqgmfm",  # seed=5
        #     # # no-norm, d=512
        #     # "cuaukeiq",  # seed=1
        #     # "n8xdctnq",  # seed=2
        #     # "afuu4r4x",  # seed=3
        #     # "a9qzumnn",  # seed=4
        #     # "drhxy0x4",  # seed=5
        #     # # norm, d=512
        #     # "w625abb9",  # seed=1
        #     # "w1ssoh9i",  # seed=2
        #     # "wtobbh3o",  # seed=3
        #     # "4sj3p04r",  # seed=4
        #     # "wtobbh3o",  # seed=5
        #     ###### Project=False
        #     # # no-norm, d=512
        #     # "egaqv4bs",  # seed=1
        #     # "0c310u03",  # seed=2
        #     # "9aafs4w2",  # seed=3
        #     # "u0h6oplr",  # seed=4
        #     # "k95boe2n",  # seed=5
        #     # # norm, d=512
        #     # "g16tux3h",  # seed=1
        #     # "82tqwdmq",  # seed=2
        #     # "msec1uul",  # seed=3
        #     # "jd6nsd17",  # seed=4
        #     # "i09us6qm",  # seed=5
        # ],
        # "walker-run": [
        #     ###### Project=True
        #     # no-norm, d=50 project=True
        #     # "mkkk0ued",  # seed=1
        #     # "3875xe79",  # seed=2
        #     # "91tzu8ic",  # seed=3
        #     # "35y9q3sl",  # seed=4
        #     # "3bsl2nk7",  # seed=5
        #     # no-norm, d=512 project=True
        #     # "mxm3wbtx",  # seed=1
        #     # "jz8qcbhj",  # seed=2
        #     # "k3fqq3ct",  # seed=3
        #     # "6ryleoei",  # seed=4
        #     # "8ojbiyw2",  # seed=5
        #     # norm, d=512 project=True
        #     # "93c29we8",  # seed=1
        #     # "kxf7upys",  # seed=2
        #     # "v6elb6zj",  # seed=3
        #     # "3rs95mmd",  # seed=4
        #     # "uub5tiz1",  # seed=5
        #     ###### Project=False
        #     # no-norm, d=512 project=False
        #     "sp2csp3v",  # seed=1
        #     "g6latprz",  # seed=2
        #     "bsvxjlhz",  # seed=3
        #     "q504eu0p",  # seed=4
        #     "tvholfxc",  # seed=5
        #     # norm, d=512 project=False
        #     "npwa34kk",  # seed=1
        #     "8vt2l9wn",  # seed=2
        #     "dnhbpv2d",  # seed=3
        #     "w65qql3j",  # seed=4
        #     "wkge5rid",  # seed=5
        # ],
        # "cheetah-run": [
        #     ##### Project=True
        #     # # no-norm, d=50
        #     # "5qubmpxz",  # seed=1
        #     # "0td7or9m",  # seed=2
        #     # "ahxjg7v2",  # seed=3
        #     # "ygdphy9l",  # seed=4
        #     # "r8f4trip",  # seed=5
        #     # # no-norm, d=512
        #     # "5zx9auk9",  # seed=1
        #     # "vpl86osp",  # seed=2
        #     # "izli85qp",  # seed=3
        #     # "tstkpt03",  # seed=4
        #     # "pynqal0q",  # seed=5
        #     # # norm, d=512
        #     # "nwusiqnt",  # seed=1
        #     # "pxbu4024",  # seed=2
        #     # "ysrel1rl",  # seed=3
        #     # "kmj3y2ww",  # seed=4
        #     # "bc6sqbdl",  # seed=5
        #     ###### Project=False
        #     # no-norm, d=512
        #     # "r540qz5u",  # seed=1
        #     # "3wuba7qz",  # seed=2
        #     # "g3wtmdr4",  # seed=3
        #     # "zx92mgg3",  # seed=4
        #     # "93h9k4a9",  # seed=5
        #     # norm, d=512
        #     # "7kg8xyj8",  # seed=1
        #     # "gujw41jb",  # seed=2
        #     # "r9cu6yvh",  # seed=3
        #     # "lh1kdlht",  # seed=4
        #     # "8gymqblt",  # seed=5
        # ],
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
