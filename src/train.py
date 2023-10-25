#!/usr/bin/env python3
import logging
import os
import pprint
import random
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.agents.utils import EarlyStopper
import torch.nn as nn
import gymnasium as gym
import hydra
import omegaconf
from omegaconf import MISSING
from gymnasium.spaces import Box, Space
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from hydra.core.config_store import ConfigStore
from src.dmc2gymnasium import DMCGym
from stable_baselines3.common.monitor import Monitor


class ActionRepeatWrapper(gym.ActionWrapper):
    def __init__(self, env, action_repeat: int):
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action):
        reward_total = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward_total += reward
            if terminated or truncated:
                break
        return obs, reward_total, terminated, truncated, info


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    max_episode_steps: int,
    action_repeat: int = 2,
    dmc_task: Optional[str] = None,
):
    # dmc = True
    # dmc = False

    def thunk():
        if capture_video:
            if dmc_task is not None:
                env = DMCGym(domain=env_id, task=dmc_task, task_kwargs={"random": seed})
                # env = PixelObservationWrapper(
                #     env,
                #     pixels_only=False,
                #     render_kwargs={"pixels": {"height": 64, "width": 64}},
                # )
            else:
                env = gym.make(
                    env_id,
                    render_mode="rgb_array",
                    max_episode_steps=max_episode_steps,
                    # **{"frame_skip": frame_skip},
                )
        else:
            if dmc_task is not None:
                env = DMCGym(domain=env_id, task=dmc_task, task_kwargs={"random": seed})
                # env = PixelObservationWrapper(
                #     env,
                #     pixels_only=False,
                #     render_kwargs={"pixels": {"height": 64, "width": 64}},
                # )
            else:
                env = gym.make(
                    env_id,
                    max_episode_steps=max_episode_steps,
                    # frame_skip=frame_skip,
                    # **{"frame_skip": frame_skip},
                )

        # env = Monitor(
        #     env,
        #     # filename=None,
        #     # allow_early_resets=True,
        #     # reset_keywords=(),
        #     # info_keywords=(),
        #     # override_existing=True,
        # )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = ActionRepeatWrapper(env=env, action_repeat=action_repeat)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_env_list(
    env_id: str,
    num_envs: int,
    seed: int,
    run_name: str,
    max_episode_steps: int,
    capture_video: bool = False,
    action_repeat: int = 2,
) -> List[gym.Env]:
    envs_list = []
    for i in range(num_envs):
        envs_list.append(
            make_env(
                env_id=env_id,
                seed=seed + i,
                idx=i,
                capture_video=capture_video,
                run_name=run_name,
                max_episode_steps=max_episode_steps,
                action_repeat=action_repeat,
            )
        )
    return envs_list


@dataclass
class AgentConfig:
    _target_: str = "src.agents.TD3"
    device: str = "cuda"


@dataclass
class DDPGConfig(AgentConfig):
    _target_: str = "src.agents.DDPG"
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
    reset_params_freq: int = 100000  # reset params after this many param updates
    # nstep: 3
    gamma: float = 0.99
    tau: float = 0.005
    device: str = "cuda"
    name: str = "DDPG"


@dataclass
class AEConfig:
    _target_: str = "src.agents.encoders.AE"
    # _partial_: bool = True
    # observation_space: Space = MISSING
    mlp_dims: List[int] = field(default_factory=lambda: [128, 128])
    latent_dim: int = 20
    act_fn = nn.ELU
    learning_rate: float = 3e-4
    batch_size: int = 128
    utd_ratio: int = 1
    tau: float = 0.005
    encoder_reset_params_freq: int = 10000  # reset enc params after X param updates
    # early_stopper: Optional[EarlyStopper] = None
    early_stopper = None
    device: str = "cuda"
    name: str = "AE"


# # @dataclass
# class AEDDPGConfig(AgentConfig):
#     _target_: str = "src.agents.LatentActorCritic"
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
#     # build_encoder_fn: Callable[[Space], src.agents.encoders.Encoder] = partial(AE)
#     # build_actor_critic_fn: Callable[[Space, Box], ActorCritic] = partial(
#     #     src.agents.DDPG, field(default_factory=DDPGConfig)
#     # )
#     device: str = "cuda"
#     name: str = "AEDDPG"


@dataclass
class AEDDPGConfig(DDPGConfig):
    _target_: str = "src.agents.AEDDPG"
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
    name: str = "AEDDPG"


@dataclass
class TD3Config(AgentConfig):
    _target_: str = "src.agents.TD3"
    mlp_dims: List[int] = field(default_factory=lambda: [256, 256])
    exploration_noise: float = 0.2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    learning_rate: float = 3e-4
    batch_size: int = 512
    num_updates: int = 1000  # 1000 is 1 update per new data
    actor_update_freq: int = 2
    # nstep: 3
    gamma: float = 0.99
    tau: float = 0.005
    device: str = "gpu"
    name: str = "TD3"


@dataclass
class AETD3Config(TD3Config):
    _target_: str = "src.agents.AETD3"
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
#     _target_: str = "src.agents.models.mlp.TransitionModel"
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
    _target_: str = "src.agents.MPPIDDPG"
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


@dataclass
class TrainConfig:
    run_name: str

    # Agent
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Env config
    env_id: str = "cartpole"
    max_episode_steps: int = 1000
    # frame_skip: int = 1
    capture_train_video: bool = False
    capture_eval_video: bool = True
    action_repeat: int = 2
    dmc_task: Optional[str] = "swingup"

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
cs.store(name="base_train", node=TrainConfig)
cs.store(group="agent", name="ddpg", node=DDPGConfig)
cs.store(group="agent", name="td3", node=TD3Config)
cs.store(group="agent", name="ae_ddpg", node=AEDDPGConfig)
cs.store(group="agent", name="ae_ddpg", node=AEDDPGConfig)

# cs.store(group="encoder", name="ae", node=AEConfig)

# cs.store(group="agent/build_encoder_fn", name="ae", node=AEConfig)
# cs.store(group="agent/build_actor_critic_fn", name="ddpg", node=DDPGConfig)
# cs.store(group="build_encoder_fn", name="build_ae_fn", node=BuildAEFnConfig)


# cs.store(group="agent", name="base_td3", node=TD3Config)
# cs.store(group="agent", name="base_vq_ddpg", node=VectorQuantizedDDPGConfig)
# cs.store(group="agent", name="base_tae_ddpg", node=TAEDDPGConfig)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def train(cfg: TrainConfig):
    import numpy as np
    import torch
    from hydra.utils import get_original_cwd
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.evaluation import evaluate_policy

    print(cfg)

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # # random.seed(random_seed)
    # # np.random.seed(random_seed)
    # # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(cfg.seed)
    # torch.manual_seed(cfg.seed)
    # # torch.cuda.manual_seed(cfg.random_seed)
    # # torch.backends.cudnn.deterministic = True
    # # torch.backends.cudnn.benchmark = False
    # np.random.seed(cfg.seed)
    # random.seed(cfg.seed)
    # torch.backends.cudnn.determinstic = True
    # torch.backends.cudnn.benchmark = False

    # device = torch.device("cuda" if torch.cuda.is_available() and cfg.device else "cpu")
    # print(f"cfg.device == gpu {cfg.device == 'gpu'}")
    # device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    cfg.device = (
        "cuda" if torch.cuda.is_available() and (cfg.device == "cuda") else "cpu"
    )
    cfg.agent.device = cfg.device
    logger.info(f"Using device: {cfg.device}")

    # Setup vectorized environment for training/evaluation
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=cfg.env_id,
                seed=cfg.seed,
                idx=0,
                capture_video=cfg.capture_train_video,
                run_name=cfg.run_name,
                max_episode_steps=cfg.max_episode_steps,
                action_repeat=cfg.action_repeat,
                dmc_task=cfg.dmc_task,
            )
        ]
    )
    envs.single_observation_space.dtype = np.float32
    # if cfg.device == "cpu":
    #     eval_envs = SubprocVecEnv(
    #         make_env_list(
    #             env_id=cfg.env_id,
    #             num_envs=5,
    #             seed=cfg.seed + 100,
    #             capture_video=cfg.capture_eval_video,
    #             run_name=cfg.run_name,
    #             max_episode_steps=cfg.max_episode_steps,
    #             action_repeat=cfg.action_repeat,
    #         )
    #     )
    # else:
    eval_envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=cfg.env_id,
                seed=cfg.seed + 100,
                idx=0,
                capture_video=cfg.capture_train_video,
                run_name=cfg.run_name,
                max_episode_steps=cfg.max_episode_steps,
                action_repeat=cfg.action_repeat,
                dmc_task=cfg.dmc_task,
            )
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    pprint.pprint(cfg_dict)

    if cfg.use_wandb:  # Initialise WandB
        import wandb

        run = wandb.init(
            project=cfg.wandb_project_name,
            # entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            # sync_tensorboard=True,
            config=cfg_dict,
            name=cfg.run_name,
            monitor_gym=cfg.monitor_gym,
            save_code=True,
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )

    rb = ReplayBuffer(
        cfg.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        # "auto",
        torch.device(cfg.device),
        handle_timeout_termination=False,
    )

    agent = hydra.utils.instantiate(
        cfg.agent,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
    )

    # start_time = time.time()
    # mean_reward, std_reward = evaluate_policy(agent, eval_envs, n_eval_episodes=10)
    # print(f"eval time multiple envs {time.time()-start_time }")

    # start_time = time.time()
    # mean_reward, std_reward = evaluate_policy(agent, eval_envs, n_eval_episodes=10)
    # print(f"eval time multiple envs second time {time.time()-start_time }")

    # TRY NOT TO MODIFY: start the game
    first_update = True
    start_time = time.time()
    obs, _ = envs.reset(seed=cfg.seed)
    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < cfg.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            # TODO set t0 properly
            actions = agent.select_action(observation=obs, eval_mode=False, t0=False)

        # Execute action in environments
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

        # Record training metrics
        if "final_info" in infos:
            for info in infos["final_info"]:
                logger.info(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                episode_length = info["episode"]["l"]

                if cfg.use_wandb:
                    wandb.log(
                        {
                            "episodic_return": info["episode"]["r"],
                            "episodic_length": info["episode"]["l"],
                            "global_step": global_step,
                            "env_step": global_step * cfg.action_repeat,
                        }
                    )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncateds):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminateds, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if "final_info" in infos:  # Update after every episode
                # print(f"episode_length {episode_length}")
                if isinstance(episode_length, np.ndarray):
                    episode_length = episode_length[0]
                # num_updates = cfg.utd_ratio * episode_length  # TODO inc. frame skip
                if first_update:
                    if cfg.pretrain_utd is not None:
                        num_new_transitions = cfg.pretrain_utd * rb.size()
                    else:
                        num_new_transitions = episode_length
                    first_update = False
                else:
                    num_new_transitions = episode_length
                logger.info(
                    f"Training agent w. {num_new_transitions} new data @ step {global_step}..."
                )
                train_metrics = agent.update(
                    replay_buffer=rb, num_new_transitions=num_new_transitions
                )
                logger.info("Finished training agent.")

                # Log training metrics
                if cfg.use_wandb:
                    train_metrics.update(
                        {
                            "global_step": global_step,
                            "SPS": int(global_step / (time.time() - start_time)),
                            "num_new_transitions": num_new_transitions,
                            "action_repeat": cfg.action_repeat,
                            "env_step": global_step * cfg.action_repeat,
                        }
                    )
                    wandb.log({"train/": train_metrics})

            if global_step % cfg.eval_every_steps == 0:
                mean_reward, std_reward = evaluate_policy(
                    agent, eval_envs, n_eval_episodes=10
                )
                # logger.info(f"Reward {mean_reward} +/- {std_reward}")
                eval_metrics = {
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "global_step": global_step,
                    "env_step": global_step * cfg.action_repeat,
                }
                if cfg.use_wandb:
                    wandb.log({"eval/": eval_metrics})

    envs.close()


if __name__ == "__main__":
    train()  # pyright: ignore
