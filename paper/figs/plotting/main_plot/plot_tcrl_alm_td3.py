# %%
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import cm
# plt.style.use('bmh')
plt.style.use('seaborn-v0_8-whitegrid')
import seaborn as sns

plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.loc'] = 'lower right'
COLORS = ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#4daf4a',]
# %%
main_envs = ['acrobot-swingup', 'cheetah-run', 'fish-swim', 'quadruped-walk', 'walker-walk', 'humanoid-walk', 'dog-walk', 'dog-run']
def plot(df, key='episode_reward'):
    envs = np.sort(df.env.unique())
    ncol = 4
    # assert envs.shape[0] % ncol == 0
    nrow = envs.shape[0] // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))

    for idx, env in enumerate(main_envs):
        data = df[df['env'] == env]
        row = idx // ncol 
        col = idx % ncol
        ax = axs[row, col]
        hue_order = data.agent.unique()

        if idx == 4:
            sns.lineplot(x='episode', y=key, data=data, errorbar=('ci', 95), hue='agent',hue_order=hue_order, 
                    palette=COLORS[:4],
                    legend='auto', ax=ax)
            ax.legend().set_title(None)
        else:
            sns.lineplot(x='episode', y=key, data=data, errorbar=('ci', 95), hue='agent', hue_order=hue_order, 
                    palette=COLORS[:4],
                    legend=False, ax=ax)
          
        ax.set_title(" ".join([ele.capitalize() for ele in env.split('-')]))
        ax.set_xlabel('Environment Steps (1e3)')
        ax.set_ylabel('Episode Return')
    plt.tight_layout()
    plt.savefig(f'alm_policy.pdf')
    plt.show()
    
# %%
# data_path = './'
# data_list = ['tcrl', 'alm', 'sac']

# df = [pd.read_csv(f'{data_path}/{algo}_main.csv') for algo in data_list]
# plot(pd.concat(df))
# %%
data_path = './'
df = [pd.read_csv(f'{data_path}/tcrl_main.csv'), 
      pd.read_csv(f'{data_path}/alm_main.csv'),
      pd.read_csv(f'{data_path}/td3_main.csv')]

plot(pd.concat(df))
# %%
