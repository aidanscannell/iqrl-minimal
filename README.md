# World Models
Train world model using lower bound on RL objective. 

## Install

### Option 1: using make
Make a virtual environment and install the dependencies with:
```sh
make .venv/bin/activate
```
Activate the environment with:
``` sh
source .venv/bin/activate
```
The [Makefile](Makefile) just runs:
``` sh
python -m venv $(VENV)
python -m pip install --upgrade pip
pip install laplace-torch==0.1a2
pip install -e ".[experiments, dev]"
```

### Option 2: pip install
Alternatively, manually install the dependencies with:
``` sh
pip install laplace-torch==0.1a2
pip install -r requirements.txt
```
We install `laplace-torch` separately due to version conflicts with `backpacpk-for-pytorch`.

### Example
Here's a short example:
```python
import src
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
