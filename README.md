# Lifelong TCRL

## Install

### AMD GPU
Install on LUMI with:
``` sh
LIFELONG_TCRL_CONTAINER_DIR=/scratch/project_462000462/lifelong-tcrl/container
mkdir  $LIFELONG_TCRL_CONTAINER_DIR
module load LUMI/22.08
module load lumi-container-wrapper
conda-containerize new --mamba --prefix $LIFELONG_TCRL_CONTAINER_DIR environment.yml
conda-containerize update $LIFELONG_TCRL_CONTAINER_DIR --post-install post-install-amd.txt
```

To run experiments add container to path with
```sh
export PATH="/scratch/project_462000462/aidan/lifelong-tcrl/container/bin:$PATH"
```
Then you can run experiments with 
``` sh
python train.py -m ++use_wandb=True agent=ddpg_ae ++agent.train_strategy=interleaved ++agent.reset_strategy=every-x-param-updates ++agent.reset_params_freq=null ++agent.utd_ratio=1,2 env=cheetah-run ++agent.temporal_consistency=True ++agent.ae_normalize=True ++capture_eval_video=False ++agent.reconstruction_loss=False ++seed=42,34,1231,123
```

## Citation
```bibtex
@article{XXX,
    title={},
    author={},
    journal={},
    year={2023}
}
```
