# iQRL: implicitly Quantized Reinforcement Learning

## Instructions
Install dependencies:
```sh
conda env create -f environment.yml
conda activate iqrl
```
Train the agent:
``` sh
python train.py +env=walker-walk
```
To log metrics with W&B:
``` sh
python train.py +env=walker-walk ++use_wandb=True
```
All tested tasks are listed in`cfgs/env`

## Install for AMD GPU
Install on LUMI with:
``` sh
IQRL_CONTAINER_DIR=/scratch/project_462000462/iqrl/container
mkdir  $IQRL_CONTAINER_DIR
module load LUMI/22.08
module load lumi-container-wrapper
conda-containerize new --mamba --prefix $IQRL_CONTAINER_DIR environment.yml
conda-containerize update $IQRL_CONTAINER_DIR --post-install post-install-amd.txt
```

To run experiments add container to path with
```sh
export PATH="/scratch/project_462000462/aidan/iqrl/container/bin:$PATH"
```
Then you can run experiments with 
``` sh
python train.py -m ++use_wandb=True agent=iqrl ++capture_eval_video=False ++seed=1,2,3,4,5
```

## Citation
```bibtex
@article{scannellSampleEfficient2024,
    title={Sample-Efficient Reinforcement Learning with Implicitly Quantized Representations},
    author={Aidan Scannell, Kalle Kujanpää, Yi Zhao, Arno Solin, Joni Pajarinen}
    journal={},
    year={2024}
}
```
