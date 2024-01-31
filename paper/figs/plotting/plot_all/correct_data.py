#%%
import pandas as pd
import numpy as np
# %%
df1 = pd.read_csv('./tcrl.csv')
# %%
# fetch quadruped_run data
#%%
tcrl_path = 'yizhao/sprl/'
tcrl_data = {
    'path': tcrl_path,
    'data': {
        'quadruped-run': ['sfwbbi87', '3bs0g2ut', '6ck2fy07', '2hmiqbct', '3gsnqcr9',],
     }
}
