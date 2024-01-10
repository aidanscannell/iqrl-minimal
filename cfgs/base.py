#!/usr/bin/env python3
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    _target_: str = "agents.TD3"
    device: str = "cuda"


@dataclass
class DDPGConfig(AgentConfig):
    _target_: str = "agents.DDPG"
    # _partial_: bool = True
    # observation_space: Space = MISSING
    # action_space: Box = MISSING
    mlp_dims: List[int] = field(default_factory=lambda: [256, 256])
    exploration_noise: float = 1.0
    # exploration_noise: float = 0.2
    policy_noise: float = 0.2
    noise_clip: float = 0.3
    learning_rate: float = 3e-4
    batch_size: int = 256
    # num_updates: int = 1000  # 1000 is 1 update per new data
    utd_ratio: int = 1  # parameter update-to-data ratio
    actor_update_freq: int = 2  # update actor less frequently than critic
    reset_params_freq: int = 40000  # reset params after this many param updates
    nstep: int = 1
    discount: float = 0.99
    tau: float = 0.005
    device: str = "cuda"
    name: str = "DDPG"


@dataclass
class AEDDPGConfig(DDPGConfig):
    _target_: str = "agents.AEDDPG"
    # Reset stuff
    reset_strategy: str = "latent_dist"  #  "latent_dist" or "every-x-param-updates"
    reset_params_freq: int = 100000  # reset params after this many param updates
    reset_threshold: float = 0.01  # reset latent (z) when changed by more than this
    memory_size: int = 10000  # mem size for calculating ||z_{mem} - e_{\phi}(x_mem)||
    # AE config
    train_strategy: str = "interleaved"  # "interleaved" or "representation-first"
    temporal_consistency: bool = False  # if True include dynamic model in encoder
    ae_learning_rate: float = 3e-4
    ae_batch_size: int = 256
    # ae_batch_size: int = 128
    # ae_num_updates: int = 1000
    ae_utd_ratio: int = 1  # encoder parameter update-to-data ratio
    ae_patience: int = 100
    ae_min_delta: float = 0.0
    latent_dim: int = 50
    ae_tau: float = 0.005
    ae_normalize: bool = True
    simplex_dim: int = 10
    # encoder_reset_params_freq: int = 10000  # reset enc params after X param updates
    name: str = "AEDDPG"
    nstep: int = 1


@dataclass
class TrainConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"agent": "ddpg"},
            # {"env": "cartpole_swingup"},
            # {
            #     "override hydra/launcher": "triton_config",  # Use slurm (on cluster) for multirun
            # "override hydra/launcher": "slurm",  # Use slurm (on cluster) for multirun
            # "override hydra/launcher": "triton_config",  # Use slurm (on cluster) for multirun
            # },
        ]
    )

    run_name: str = f"cartpole_DDPG_42_{time.time()}"

    # Agent
    # agent: AgentConfig = field(default_factory=AgentConfig)
    # env: EnvConfig = field(default_factory=CartpoleSwingupEnvConfig)

    # Env config
    env_id: str = "cartpole"
    dmc_task: Optional[str] = None
    # frame_skip: int = 1

    # Experiment config
    max_episode_steps: int = 1000  # Max episode length
    num_episodes: int = 500  # Number of training episodes
    random_episodes: int = 10  # Number of random episodes at start
    pretrain_utd: Optional[int] = None  # If not none pretrain on random data with utd
    action_repeat: int = 2
    buffer_size: int = int(1e6)
    seed: int = 42
    device: str = "cuda"  # "cpu" or "cuda" etc

    # Evaluation
    eval_every_episodes: int = 2
    num_eval_episodes: int = 10
    capture_train_video: bool = False
    capture_eval_video: bool = True

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = ""
    monitor_gym: bool = True


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
# cs.store(group="agent", name="ddpg", node=DDPGConfig)
# cs.store(group="agent", name="td3", node=TD3Config)
# cs.store(group="agent", name="ae_ddpg", node=AEDDPGConfig)
