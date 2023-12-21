#!/usr/bin/env python3
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from hydra.core.config_store import ConfigStore


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
    exploration_noise: float = 0.2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    learning_rate: float = 3e-4
    batch_size: int = 512
    # num_updates: int = 1000  # 1000 is 1 update per new data
    utd_ratio: int = 1  # parameter update-to-data ratio
    actor_update_freq: int = 1  # update actor less frequently than critic
    reset_params_freq: int = 40000  # reset params after this many param updates
    # nstep: 3
    discount: float = 0.99
    tau: float = 0.005
    device: str = "cuda"
    name: str = "DDPG"


@dataclass
class SACConfig(AgentConfig):
    _target_: str = "agents.SAC"
    # _partial_: bool = True
    # observation_space: Space = MISSING
    # action_space: Box = MISSING
    mlp_dims: List[int] = field(default_factory=lambda: [256, 256])
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    linear_approx: bool = False
    alpha: float = 0.2  # entropy regularization coefficient
    autotune: bool = True  # automatic tuning of the entropy coefficient
    exploration_noise: float = 0.2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    q_lr: float = 1e-3
    policy_lr: float = 3e-4
    batch_size: int = 32
    # num_updates: int = 1000  # 1000 is 1 update per new data
    utd_ratio: int = 1  # parameter update-to-data ratio
    actor_update_freq: int = 1  # update actor less frequently than critic
    reset_params_freq: int = 40000  # reset params after this many param updates
    # nstep: 3
    discount: float = 0.99
    tau: float = 0.005
    device: str = "cuda"
    name: str = "SAC"


# @dataclass
# class AEConfig:
#     _target_: str = "agents.encoders.AE"
#     # _partial_: bool = True
#     # observation_space: Space = MISSING
#     mlp_dims: List[int] = field(default_factory=lambda: [128, 128])
#     latent_dim: int = 20
#     act_fn = nn.ELU
#     learning_rate: float = 3e-4
#     batch_size: int = 128
#     utd_ratio: int = 1
#     tau: float = 0.005
#     encoder_reset_params_freq: int = 10000  # reset enc params after X param updates
#     # early_stopper: Optional[EarlyStopper] = None
#     early_stopper = None
#     device: str = "cuda"
#     name: str = "AE"


# # @dataclass
# class AEDDPGConfig(AgentConfig):
#     _target_: str = "agents.LatentActorCritic"
#     defaults: List[Any] = field(
#         default_factory=lambda: [
#             {
#                 "build_encoder_fn": "ae",
#                 # "build_actor_critic_fn": "ddpg",
#             }
#         ]
#     )
#     # build_encoder_fn = MISSING
#     # build_actor_critic_fn = MISSING
#     # build_encoder_fn: Callable[[Space], agents.encoders.Encoder] = partial(AE)
#     # build_actor_critic_fn: Callable[[Space, Box], ActorCritic] = partial(
#     #     agents.DDPG, field(default_factory=DDPGConfig)
#     # )
#     device: str = "cuda"
#     name: str = "AEDDPG"


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
    ae_batch_size: int = 512
    # ae_batch_size: int = 128
    # ae_num_updates: int = 1000
    ae_utd_ratio: int = 1  # encoder parameter update-to-data ratio
    ae_patience: int = 100
    ae_min_delta: float = 0.0
    latent_dim: int = 20
    ae_tau: float = 0.005
    ae_normalize: bool = True
    simplex_dim: int = 10
    # encoder_reset_params_freq: int = 10000  # reset enc params after X param updates
    name: str = "AEDDPG"


@dataclass
class VQDDPGConfig(DDPGConfig):
    _target_: str = "agents.VQDDPG"
    # Reset stuff
    reset_strategy: str = "latent_dist"  #  "latent_dist" or "every_x_param_updates"
    reset_params_freq: int = 100000  # reset params after this many param updates
    reset_threshold: float = 0.01
    memory_size: int = 10000
    # AE config
    ae_learning_rate: float = 3e-5
    ae_batch_size: int = 128
    # ae_num_updates: int = 1000
    ae_utd_ratio: int = 1  # encoder parameter update-to-data ratio
    ae_patience: int = 100
    ae_min_delta: float = 0.0
    levels: List[int] = field(default_factory=lambda: [8, 2, 2])
    # levels: List[int] = field(default_factory=lambda: [8, 6, 5])
    ae_tau: float = 0.005
    # encoder_reset_params_freq: int = 10000  # reset enc params after X param updates
    name: str = "VQDDPG"


@dataclass
class TD3Config(AgentConfig):
    _target_: str = "agents.TD3"
    mlp_dims: List[int] = field(default_factory=lambda: [256, 256])
    exploration_noise: float = 0.2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    learning_rate: float = 3e-4
    batch_size: int = 512
    num_updates: int = 1000  # 1000 is 1 update per new data
    actor_update_freq: int = 2
    # nstep: 3
    discount: float = 0.99
    tau: float = 0.005
    device: str = "gpu"
    name: str = "TD3"


@dataclass
class AETD3Config(TD3Config):
    _target_: str = "agents.AETD3"
    # AE config
    ae_learning_rate: float = 3e-4
    ae_batch_size: int = 128
    # ae_num_updates: int = 1000
    ae_utd_ratio: int = 1  # encoder parameter update-to-data ratio
    ae_patience: int = 100
    ae_min_delta: float = 0.0
    latent_dim: int = 20
    ae_tau: float = 0.005
    encoder_reset_params_freq: int = 10000  # reset enc params after X param updates
    name: str = "AETD3"


# @dataclass
# class TransitionModelConfig:
#     _target_: str = "agents.models.mlp.TransitionModel"
#     # observation_space: Space
#     # action_space: Box
#     mlp_dims: List[int] = field(default_factory=lambda: [256, 256])
#     act_fn = nn.ELU
#     learning_rate: float = 3e-4
#     batch_size: int = 128
#     utd_ratio: int = 1  # transition model parameter update-to-data ratio
#     early_stopper: EarlyStopper = None
#     device: str = "cuda"


@dataclass
class MPPIDDPGConfig(DDPGConfig):
    _target_: str = "agents.MPPIDDPG"
    horizon: int = 5
    num_mppi_iterations: int = 5
    num_samples: int = 512
    mixture_coef: float = 0.05
    num_topk: int = 64
    temperature: float = 0.5
    momentum: float = 0.1
    unc_prop_strategy: str = "mean"  # "mean" or "sample", "sample" require transition_model to use SVGP prediction type
    sample_actor: bool = True
    bootstrap: bool = True
    device: str = "cuda"
    name: str = "MPPI"


# @dataclass
# class TritonConfig:
#     defaults: List[Any] = field(
#         default_factory=lambda: [{"submitit_slurm": "submitit_slurm"}]
#     )

#     _target_: str = (
#         "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
#     )
#     # submitit_folder: ${hydra.sweep.dir}/.submitit/%j
#     timeout_min: int = 120  # 2 hours
#     tasks_per_node: int = 1
#     nodes: int = 1
#     # name: str = 1  # ${hydra.job.name}
#     comment: Optional[str] = None
#     exclude: Optional[str] = None
#     signal_delay_s: int = 600
#     max_num_timeout: int = 20
#     # additional_parameters: dict = {}
#     array_parallelism: int = 256
#     # setup: List = []
#     constraint: str = "volta"
#     mem_gb: int = 50
#     gres: str = "gpu:1"


# @dataclass
# class EnvConfig:
#     env_id: str
#     dmc_task: Optional[str] = None


# @dataclass
# class CartpoleSwingupEnvConfig(EnvConfig):
#     env_id: str = "cartpole"
#     dmc_task: Optional[str] = "swingup"


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
    max_episode_steps: int = 1000
    # frame_skip: int = 1
    capture_train_video: bool = False
    capture_eval_video: bool = True
    action_repeat: int = 2

    # Experiment config
    exp_name: str = "base"
    buffer_size: int = int(1e6)
    pretrain_utd: Optional[int] = None  # If not none pretrain on random data with utd
    learning_starts: int = int(25e3)
    total_timesteps: int = int(1e6)
    logging_epoch_freq: int = 100
    eval_every_steps: int = 2500
    seed: int = 42
    device: str = "cuda"  # "cpu" or "cuda" etc
    # cuda: bool = True  # if gpu available put on gpu
    debug: bool = False
    # torch_deterministic: bool = True

    # W&B config
    wandb_project_name: str = ""
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    use_wandb: bool = False
    monitor_gym: bool = True


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
# cs.store(name="triton_config", group="hydra/launcher", node=TritonConfig)
cs.store(group="agent", name="ddpg", node=DDPGConfig)
cs.store(group="agent", name="td3", node=TD3Config)
cs.store(group="agent", name="ae_ddpg", node=AEDDPGConfig)
cs.store(group="agent", name="vq_ddpg", node=VQDDPGConfig)

# cs.store(group="env", name="env_config", node=EnvConfig)
# cs.store(group="env", name="cartpole_swingup", node=CartpoleSwingupEnvConfig)

# cs.store(group="encoder", name="ae", node=AEConfig)

# cs.store(group="agent/build_encoder_fn", name="ae", node=AEConfig)
# cs.store(group="agent/build_actor_critic_fn", name="ddpg", node=DDPGConfig)
# cs.store(group="build_encoder_fn", name="build_ae_fn", node=BuildAEFnConfig)


# cs.store(group="agent", name="base_td3", node=TD3Config)
# cs.store(group="agent", name="base_vq_ddpg", node=VectorQuantizedDDPGConfig)
# cs.store(group="agent", name="base_tae_ddpg", node=TAEDDPGConfig)
