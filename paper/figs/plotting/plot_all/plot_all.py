# %%
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import cm

# plt.style.use('bmh')
plt.style.use("seaborn-v0_8-whitegrid")
import seaborn as sns

plt.rcParams["figure.dpi"] = 400
plt.rcParams["font.size"] = 13
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["legend.loc"] = "lower right"

# %%
# first process SAC data
COLORS = {
    "TCRL": "#e41a1c",
    "SAC": "#377eb8",
    "SAC-our": "#984ea3",
    "VQ-TD3": "magenta",
}


def plot(df, key="episode_reward"):
    envs = np.sort(env_list)
    ncol = 4
    # assert envs.shape[0] % ncol == 0
    nrow = envs.shape[0] // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))

    for idx, env in enumerate(envs):
        data = df[df["env"] == env]
        data = data[data["episode"] <= max_ep[env]]
        row = idx // ncol
        col = idx % ncol
        ax = axs[row, col]
        hue_order = data.agent.unique()

        if idx == 0:
            sns.lineplot(
                # x="episode",
                x="env_step",
                y=key,
                data=data,
                errorbar=("ci", 95),
                hue="agent",
                hue_order=hue_order,
                palette=COLORS,
                legend="auto",
                ax=ax,
            )
            ax.legend().set_title(None)
        else:
            sns.lineplot(
                x="env_step",
                # x="episode",
                y=key,
                data=data,
                errorbar=("ci", 95),
                hue="agent",
                hue_order=hue_order,
                palette=COLORS,
                legend=False,
                ax=ax,
            )

        ax.set_title(" ".join([ele.capitalize() for ele in env.split("-")]))
        ax.set_xlabel("Environment Steps (1e3)")
        ax.set_ylabel("Episode Return")
    plt.tight_layout()
    plt.savefig(f"all_policy_new.pdf")
    # plt.show()


# %%
data_path = "./"
data_list = ["tcrl", "sac", "sac_our", "vq_td3"]

# read tcrl's data to get the env_list and max_episode
df = pd.read_csv("./tcrl.csv")
env_list = df["env"].unique()
max_ep = {env: df[df["env"] == env]["episode"].max() for env in env_list}

df = [pd.read_csv(f"{data_path}/{algo}.csv") for algo in data_list]
plot(pd.concat(df))
# %%
