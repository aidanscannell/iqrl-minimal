# %%
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib import cm

# plt.style.use('bmh')
plt.style.use("seaborn-v0_8-whitegrid")
import seaborn as sns

plt.rcParams["figure.dpi"] = 400
plt.rcParams["font.size"] = 13
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["legend.loc"] = "lower right"
plt.rcParams["text.usetex"] = True
COLORS = {
    # "TCRL": "#e41a1c",
    # "SAC": "#377eb8",
    # "REDQ": "#984ea3",
    # "TD-MPC": "#ff7f00",
    # "VQ-TD3": "magenta",
    "iQRL": "#e41a1c",
    "iQRL+rew": "#377eb8",
    "iQRL+cos+rew": "#ff7f00",
    "TCRL-ours": "#4DAF4A",
    # "TCRL-ours-small": "#984ea3",
}
# %%
main_envs = [
    "cheetah-run",
    "acrobot-swingup",
    # "walker-walk",
    "walker-run",
    "hopper-stand",
    "fish-swim",
    "quadruped-run",
    # "humanoid-walk",
    # "humanoid-run",
    # "hopper-hop",
    # "dog-walk",
    # "dog-run",
]

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


def plot(df, key="episode_reward"):
    # breakpoint()
    envs = np.sort(df.env.unique())
    ncol = 3
    # assert envs.shape[0] % ncol == 0
    # nrow = len(main_envs) // ncol
    nrow = 2
    # nrow = envs.shape[0] // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))

    df["episode"] = df.apply(lambda row: int(row["env_step"] / 1000), axis=1)

    df = df.rename(columns=rename)

    for idx, env in enumerate(main_envs):
        data = df[df["env"] == env]
        row = idx // ncol
        col = idx % ncol
        # ax = axs[row, col]
        ax = axs[row, col]
        # hue_order = data.agent.unique()
        hue_order = data.name.unique()
        # hue_order = data.utd_ratio.unique()
        # breakpoint()

        if idx == 0:
            sns.lineplot(
                # x="env_step",
                x="episode",
                y=key,
                data=data,
                errorbar=("ci", 95),
                hue="name",
                # hue_order=hue_order,
                palette=COLORS,
                legend="auto",
                ax=ax,
            )
            ax.legend().set_title(None)
        else:
            sns.lineplot(
                # x="env_step",
                x="episode",
                y=key,
                data=data,
                errorbar=("ci", 95),
                hue="name",
                # hue_order=hue_order,
                palette=COLORS,
                legend=False,
                ax=ax,
            )

        #        if idx == 0:
        #            print(data)
        #            print(ax)
        #            exit(0)
        #            sns.lineplot(
        #                x="env_step",
        #                # x="episode",
        #                y=key,
        #                data=data,
        #                errorbar=("ci", 95),
        #                hue="name",
        #                # style="utd_ratio",
        #                hue_order=hue_order,
        #                palette=COLORS,
        #                legend="auto",
        #                ax=ax,
        #            )
        #            ax.legend().set_title(None)
        #        else:
        #            # breakpoint()
        #            sns.lineplot(
        #                x="env_step",
        #                # x="episode",
        #                y=key,
        #                data=data,
        #                errorbar=("ci", 95),
        #                hue="name",
        #                # style="utd_ratio",
        #                hue_order=hue_order,
        #                palette=COLORS,
        #                legend=False,
        #                ax=ax,
        #            )

        ax.set_title(" ".join([ele.capitalize() for ele in env.split("-")]))
        ax.set_xlabel("Environment Steps (1e3)")
        ax.set_ylabel("Episode Return")
    plt.tight_layout()
    plt.savefig(f"../../reward-ablation.pdf")
    # plt.show()


# %%
# data_path = './'
# data_list = ['tcrl', 'alm', 'sac']

# df = [pd.read_csv(f'{data_path}/{algo}_main.csv') for algo in data_list]
# plot(pd.concat(df))
# %%
# process redq data
data_file_path = "./iqrl.csv"
assert os.path.isfile(data_file_path)
# %%
#
df = [
    pd.read_csv(data_file_path),
]
plot(pd.concat(df))

# %%
