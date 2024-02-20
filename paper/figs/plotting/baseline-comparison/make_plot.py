# %%
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import cm

# plt.style.use('bmh')
plt.style.use("seaborn-v0_8-whitegrid")
import seaborn as sns

plt.rcParams["figure.dpi"] = 400
plt.rcParams["font.size"] = 15
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["legend.loc"] = "lower right"
plt.rcParams["text.usetex"] = True
COLORS = {
    "TCRL": "#4daf4a",
    "SAC": "#377eb8",
    "REDQ": "#984ea3",
    "TD-MPC": "#ff7f00",
    "iQRL": "#e41a1c",
    # "iFSQ-RL d=1024": "magenta",
    # "iFSQ-RL d=512": "grey",
    # "iFSQ-RL d=128": "#e41a1c",
}
# %%
main_envs = [
    "acrobot-swingup",
    "cheetah-run",
    "fish-swim",
    "quadruped-walk",
    # "humanoid-run",
    "walker-walk",
    "humanoid-walk",
    "dog-walk",
    # "dog-run",
]


def plot(df, key="episode_reward"):
    envs = np.sort(df.env.unique())
    ncol = 4
    # assert envs.shape[0] % ncol == 0
    # nrow = len(main_envs) // ncol
    nrow = 2
    # nrow = envs.shape[0] // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))

    df["episode"] = df.apply(lambda row: int(row["env_step"] / 1000), axis=1)

    for idx, env in enumerate(main_envs):
        idx += 1
        data = df[df["env"] == env]
        row = idx // ncol
        col = idx % ncol
        ax = axs[row, col]
        hue_order = data.agent.unique()

        if idx == 2:
            g = sns.lineplot(
                # x="env_step",
                x="episode",
                y=key,
                data=data,
                errorbar=("ci", 95),
                hue="agent",
                # hue_order=hue_order,
                palette=COLORS,
                legend="auto",
                ax=ax,
            )
            ax.legend().set_title(None)
        else:
            g = sns.lineplot(
                # x="env_step",
                x="episode",
                y=key,
                data=data,
                errorbar=("ci", 95),
                hue="agent",
                # hue_order=hue_order,
                palette=COLORS,
                legend=False,
                ax=ax,
            )
        if env == "quadruped-walk":
            g.set(xlim=(0, 500))
        if env == "walker-walk":
            g.set(xlim=(0, 250))
        if env == "dog-walk":
            g.set(xlim=(0, 750))
        if env == "humanoid-walk":
            g.set(xlim=(0, 3000))

        ax.set_title(" ".join([ele.capitalize() for ele in env.split("-")]))
        ax.set_xlabel("Environment Steps (1e3)")
        ax.set_ylabel("Episode Return")

    # df["episode_reward"] = df.apply(
    #     lambda row: int(row["episode_reward"] / len(main_envs)), axis=1
    # )

    # Only plot for first 1M steps
    df = df[df["episode"] < 1000]
    # df = df[df["env"] != "walker-walk"]
    g = sns.lineplot(
        # x="env_step",
        x="episode",
        y=key,
        data=df,
        errorbar=("ci", 95),
        hue="agent",
        # hue_order=hue_order,
        palette=COLORS,
        legend=False,
        ax=axs[0, 0],
    )
    axs[0, 0].set_title("Avg. Over 7 DMC Tasks")
    axs[0, 0].set_xlabel("Environment Steps (1e3)")
    axs[0, 0].set_ylabel("Episode Return")

    plt.tight_layout()
    plt.savefig(f"../../baselines_comparison_debug.pdf")
    # plt.savefig(f"../../baselines_comparison.pdf")
    # plt.show()


# %%
# data_path = './'
# data_list = ['tcrl', 'alm', 'sac']

# df = [pd.read_csv(f'{data_path}/{algo}_main.csv') for algo in data_list]
# plot(pd.concat(df))
# %%
# process redq data
data_path = "./"
# redq with utd 10 fails to solve fish-swim, thus we change it to 1 for this task
df_redq_utd10 = pd.read_csv(f"{data_path}/redq_utd10_main.csv")
df_redq_utd1 = pd.read_csv(f"{data_path}/redq_utd1_main.csv")

df_redq = pd.concat(
    [
        df_redq_utd10[df_redq_utd10["env"] != "fish-swim"],
        df_redq_utd1[df_redq_utd1["env"] == "fish-swim"],
    ]
)

# %%

df = [
    pd.read_csv(f"{data_path}/tcrl_main.csv"),
    pd.read_csv(f"{data_path}/tdmpc_main.csv"),
    df_redq,
    pd.read_csv(f"{data_path}/sac_main.csv"),
    pd.read_csv(f"{data_path}/iqrl.csv"),
]
plot(pd.concat(df))

# %%
