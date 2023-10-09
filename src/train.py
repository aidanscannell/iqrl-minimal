#!/usr/bin/env python3
import logging
import os
import pprint
import random
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gymnasium as gym
import hydra
import numpy as np
import omegaconf
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    max_episode_steps: int,
    frame_skip: int = 5,
):
    def thunk():
        if capture_video:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                max_episode_steps=max_episode_steps,
                # **{"frame_skip": frame_skip},
            )
        else:
            env = gym.make(
                env_id,
                max_episode_steps=max_episode_steps,
                # frame_skip=frame_skip,
                # **{"frame_skip": frame_skip},
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
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
            )
        )
    return envs_list


@dataclass
class AgentConfig:
    _target_: str = "src.agents.TD3"


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


@dataclass
class TrainConfig:
    run_name: str

    # Agent
    agent: AgentConfig = field(default_factory=TD3Config)
    utd_ratio: int = 1

    # Env config
    env_id: str = "CartPole"
    max_episode_steps: int = 1000
    # frame_skip: int = 1
    capture_train_video: bool = False
    capture_eval_video: bool = True

    # Experiment config
    exp_name: str = "base"
    buffer_size: int = int(1e6)
    learning_starts: int = int(25e3)
    total_timesteps: int = int(1e6)
    logging_epoch_freq: int = 100
    eval_every_steps: int = 5000
    seed: int = 42
    device: str = "cpu"  # "cpu" or "gpu" etc
    # cuda: bool = True  # if gpu available put on gpu
    debug: bool = False
    torch_deterministic: bool = True

    # W&B config
    wandb_project_name: str = ""
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    use_wandb: bool = False
    monitor_gym: bool = True


cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)
cs.store(group="agent", name="base_td3", node=TD3Config)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def train(cfg: TrainConfig):
    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # device = torch.device("cuda" if torch.cuda.is_available() and cfg.device else "cpu")
    # print(f"cfg.device == gpu {cfg.device == 'gpu'}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.device == "gpu" else "cpu"
    )

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
            )
        ]
    )
    envs.single_observation_space.dtype = np.float32
    eval_envs = SubprocVecEnv(
        make_env_list(
            env_id=cfg.env_id,
            num_envs=5,
            seed=cfg.seed + 1,
            capture_video=cfg.capture_eval_video,
            run_name=cfg.run_name,
            max_episode_steps=cfg.max_episode_steps,
        )
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
        device,
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
                num_updates = cfg.utd_ratio * episode_length[0]  # TODO inc. frame skip
                logger.info(
                    f"Training agent w. {num_updates} updates @ step {global_step}..."
                )
                train_metrics = agent.train(replay_buffer=rb, num_updates=num_updates)
                logger.info("Finished training agent.")

                # Log training metrics
                if cfg.use_wandb:
                    train_metrics.update(
                        {
                            "global_step": global_step,
                            "SPS": int(global_step / (time.time() - start_time)),
                            "num_updates": num_updates,
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
                }
                if cfg.use_wandb:
                    wandb.log({"eval/": eval_metrics})

    envs.close()


if __name__ == "__main__":
    train()  # pyright: ignore
