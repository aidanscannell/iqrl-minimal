import pathlib

import setuptools

_here = pathlib.Path(__file__).resolve().parent
print(_here)

name = "src"
author = ""
author_email = ""
description = "World models for RL in PyTorch."

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = ""

license = "Apache-2.0"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]
keywords = [
    "deep-learning",
    "machine-learning",
    "reinforcement-learning",
    "model-based-reinforcement-learning",
    "planning",
]

python_requires = ">=3.8"

install_requires = [
    "torch",
    "torchvision",
    "torchtyping",
    "matplotlib",
    "numpy",
    "stable_baselines3",
    "gymnasium",
    "imageio",
    "umap",
    "vector_quantize_pytorch",
]
extras_require = {
    "dev": [
        "black==23.3.0",
        "pre-commit==3.2.2",
        "pyright==1.1.301",
        "isort==5.12.0",
        "pyflakes==3.0.1",
        "pytest==7.2.2",
    ],
    "experiments": [
        "wandb",
        "hydra-core",
        "hydra-submitit-launcher",
        "jupyter",
        "mujoco==2.3.3",
        "dm_control==1.0.11",  # deepmind control suite
        "urllib3>=1.26.11",  # TODO check this works
        "moviepy",
        # "opencv-python==4.7.0.72",
        # "moviepy==1.0.3",  # rendering
        # "tikzplotlib",
        # "tikzplotlib==0.10.1",
        # "gpytorch==1.9.1",  # for RL SVGP experiments
        # "gym[classic_control]==0.26.2",
        # "pandas",  # for making UCI table
        # "seaborn",
        # "plotly==5.1.0",
    ],
}

setuptools.setup(
    name=name,
    version="0.1.0",
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    keywords=keywords,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    # packages=setuptools.find_namespace_packages(),
    # packages=setuptools.find_packages(),  # packages=setuptools.find_packages(exclude=["notebooks", "paper"]),
    # package_dir = {"": "src"},
    packages=setuptools.find_packages(exclude=["notebooks"]),
)
