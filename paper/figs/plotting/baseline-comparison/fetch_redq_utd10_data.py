#%%
redq_path = 'yizhao/redq_aligned/'
redq_utd10_data = {
    'path': redq_path,
    'data': {
        'acrobot-swingup': ['2avs8qvo','wy9mr8ke','2tcw6uat','xk9hhggf','240610cf'],
        'cheetah-run': ['3bykfzb8','3t5b2qbg','2xwmvfx1','1bhfo6ql','1sp0l2mc'],
        'fish-swim': ['2crog1zw','21cobt86','1omunkwt','2mfay1qd','2mym8cu9'],
        'quadruped-walk': ['2gmklz3o','2fa7fmvq','2lowi443','2mpy5njk','1saqkms5'],
        'walker-walk': ['19p8pd8a','2v3j1ks2','28drab7y','3r7xgjq5','1h2uhoeg'],
        'humanoid-walk': ['26x8gnsw','2lif0urn','2nt23llu','1f150tks'],
        'dog-walk': ['2d1g8awy','2jqf95ei','2dnna3pd','2gxvg107','2e7wc6u3', '2ef2ejbb'],
        'dog-run': ['l0jxplhr','31sbxbbt','3vsqdij1','17185yhl','10xfrxc7', '2ipeutzk'],
    }
}

# %%
# Plot wandb
import numpy as np
from typing import Dict

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import wandb
api = wandb.Api(timeout=19)
import pandas as pd

# pet-pytorch
titles = ['env_step','episode','episode_reward', 'time',]
keys = ['eval/.env_step', 'eval/.episode', 'eval/.episode_reward', 'eval/.total_time']

rename = {a : b for a, b in zip(keys, titles)}

def fetch_results(run_path, run_name_list, keys, agent_name):

    data = []
    for run_name in run_name_list:
        wandb_run = api.run(run_path + run_name)
        history = wandb_run.history(keys=keys, pandas=True)
        history = history.rename(columns=rename)
        # obtain seed, and env_name
        seed, env_name = wandb_run.config['_content']['seed'], wandb_run.config['_content']['env']

        # append env_name, seed, and agent_name
        history['env'] = env_name.replace('_', '-', 1)
        history['seed'] = seed
        history['agent'] = agent_name

        data.append(history)
    return pd.concat(data)

# %%
data = pd.concat([fetch_results(run_path=redq_utd10_data['path'], 
                    run_name_list=redq_utd10_data['data'][env], 
                    keys=keys,
                    agent_name="REDQ")
        for env in redq_utd10_data['data'].keys()])

#%%

data.to_csv('./redq_utd10_main.csv')


# %%
