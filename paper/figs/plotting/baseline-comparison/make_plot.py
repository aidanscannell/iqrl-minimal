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
COLORS = {
    "TCRL": "#4daf4a",
    "SAC": "#377eb8",
    "REDQ": "#984ea3",
    "TD-MPC": "#ff7f00",
    "iFSQ-RL": "#e41a1c",
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
    "dog-run",
]


def plot(df, key="episode_reward"):
    envs = np.sort(df.env.unique())
    ncol = 4
    # assert envs.shape[0] % ncol == 0
    nrow = len(main_envs) // ncol
    # nrow = envs.shape[0] // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))

    # df["env_step"] = df["env_step"] / 1000
    for idx, env in enumerate(main_envs):
        data = df[df["env"] == env]
        # breakpoint()
        # data[data["agent"] == "iFSQ-RL"] = data[data["agent"] == "iFSQ-RL"].iloc[::2]
        row = idx // ncol
        col = idx % ncol
        ax = axs[row, col]
        hue_order = data.agent.unique()

        if idx == 4:
            sns.lineplot(
                # x=int("env_step" / 1000),
                x="env_step",
                # x="env_step",
                # x="episode",
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
                # x="episode",
                x="env_step",
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
    plt.savefig(f"../../baselines_comparison.pdf")
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
    pd.read_csv(f"{data_path}/vq_td3_main.csv"),
]
plot(pd.concat(df))

# %%
