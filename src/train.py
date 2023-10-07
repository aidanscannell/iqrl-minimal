import logging
import os
import pprint
import random
import shutil
import time
from dataclasses import dataclass
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


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


@dataclass
class TrainConfig:
    observation_space_shape: Tuple
    action_space_shape: Tuple
    action_space_low: int
    action_space_high: int

    run_name: str
    env_id: str = "CartPole"
    learning_starts: int = int(25e3)
    total_timesteps: int = int(1e6)

    # Agent stuff
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    policy_frequency: int = 2
    noise_clip: float = 0.5
    tau: float = 0.005  # target smoothing coefficient
    gamma: float = 0.99  # discount factor
    # Training config
    batch_size: int = 64
    learning_rate: float = 3e-4
    buffer_size: int = int(1e6)

    # Experiment config
    exp_name: str = "base"
    logging_epoch_freq: int = 100
    seed: int = 42
    device: str = "cpu"  # "cpu" or "gpu" etc
    # cuda: bool = True  # if gpu available put on gpu
    capture_train_video: bool = True
    capture_eval_video: bool = True
    debug: bool = False
    torch_deterministic: bool = True

    # W&B config
    wandb_project_name: str = ""
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    use_wandb: bool = False
    monitor_gym: bool = True


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def train(cfg: TrainConfig):
    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # device = torch.device("cuda" if torch.cuda.is_available() and cfg.device else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() and cfg.device else "cpu")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.device is "gpu" else "cpu"
    )
    # cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup vectorized environment for training/evaluation
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(cfg.env_id, cfg.seed, 0, cfg.capture_train_video, cfg.run_name),
            # make_env(
            #     cfg.env_id, cfg.seed + 1, 1, cfg.capture_train_video, cfg.run_name
            # ),
        ]
    )
    eval_envs = gym.vector.SyncVectorEnv(
        # [make_env(cfg.env_id, cfg.seed + 1, 0, cfg.capture_eval_video, cfg.run_name)]
        [
            make_env(cfg.env_id, cfg.seed + 2, 0, cfg.capture_eval_video, cfg.run_name),
            # make_env(cfg.env_id, cfg.seed + 3, 1, cfg.capture_eval_video, cfg.run_name),
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # Initialise WandB
    if cfg.use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project_name,
            # entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            # sync_tensorboard=True,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            name=cfg.run_name,
            monitor_gym=cfg.monitor_gym,
            save_code=True,
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        try:
            shutil.copytree(
                os.path.abspath(".hydra"),
                os.path.join(os.path.join(get_original_cwd(), run.dir), "hydra"),
            )
            wandb.save("hydra")
        except FileExistsError:
            pass

    cfg_dict = (
        omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    pprint.pprint(cfg_dict)
    # logger.info(f"Config: {cfg}")
    # cfg.run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"  # TODO move to cfg

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        cfg.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    agent = hydra.utils.instantiate(
        cfg.agent,
        observation_space_shape=envs.single_observation_space.shape,
        action_space_shape=envs.single_action_space.shape,
        action_space_low=envs.single_action_space.low,
        action_space_high=envs.single_action_space.high,
    )

    class EvalAgentWrapper:
        def predict(
            self, observation, state=None, episode_start=None, deterministic=False
        ):
            action = agent.select_action(observation=observation, eval_mode=True)
            recurrent_state = None
            return action, recurrent_state

    from stable_baselines3.common.evaluation import evaluate_policy

    eval_agent = EvalAgentWrapper()
    mean_reward, _ = evaluate_policy(eval_agent, eval_envs, n_eval_episodes=10)
    print(f"MEAN_REWARD {mean_reward}")

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
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
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
                t = 0
        rb.add(obs, real_next_obs, actions, rewards, terminateds, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            # if "final_info" in infos:
            if global_step % 1000 == 0:
                logger.info(f"Training agent step {global_step}...")
                train_metrics = agent.train(replay_buffer=rb, global_step=global_step)
                logger.info("Finished training agent.")

                # Log training metrics
                if cfg.use_wandb:
                    train_metrics.update(
                        {
                            "global_step": global_step,
                            "SPS": int(global_step / (time.time() - start_time)),
                        }
                    )
                    wandb.log({"train/": train_metrics})
            if global_step % 5000 == 0:
                mean_reward, std_reward = evaluate_policy(
                    eval_agent, eval_envs, n_eval_episodes=10
                )
                logger.info(f"Reward {mean_reward} +/- {std_reward}")
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
