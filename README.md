# World Models
Train world model using lower bound on RL objective. 

## Install

### AMD GPU
Install on LUMI with:
``` sh
WORLD_MODELS_CONTAINER_DIR=/scratch/project_462000217/aidan/world-models/container
mkdir  $WORLD_MODELS_CONTAINER_DIR
module load LUMI/22.08
module load lumi-container-wrapper
conda-containerize new --mamba --prefix $WORLD_MODELS_CONTAINER_DIR environment.yml
conda-containerize update $WORLD_MODELS_CONTAINER_DIR --post-install post-install-amd.txt
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
