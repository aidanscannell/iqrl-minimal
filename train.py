#!/usr/bin/env python3
import os

import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd


# from cfgs.base import TrainConfig


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train")
def train(cfg):
    # def train(cfg: TrainConfig):
    import logging
    import pprint
    import random
    import time
    from functools import partial

    import gymnasium as gym
    import numpy as np
    import torch
    from utils.buffers import ReplayBuffer
    from stable_baselines3.common.evaluation import evaluate_policy
    from utils.env import make_env

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    ###### Fix seed for reproducibility ######
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    cfg.device = (
        "cuda" if torch.cuda.is_available() and (cfg.device == "cuda") else "cpu"
    )
    cfg.agent.device = cfg.device
    logger.info(f"Using device: {cfg.device}")

    ###### Setup vectorized environment for training/evaluation ######
    make_env_fn = partial(
        make_env,
        env_id=cfg.env_id,
        idx=0,
        run_name=cfg.run_name,
        max_episode_steps=cfg.max_episode_steps,
        action_repeat=cfg.action_repeat,
        dmc_task=cfg.dmc_task,
    )
    envs = gym.vector.SyncVectorEnv(
        [make_env_fn(seed=cfg.seed, capture_video=cfg.capture_train_video)]
    )
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env_fn(seed=cfg.seed + 100, capture_video=cfg.capture_eval_video)]
    )
    envs.single_observation_space.dtype = np.float32
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    ###### Initialise W&B ######
    if cfg.use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project_name,
            group=f"{cfg.env_id}-{cfg.dmc_task}",
            tags=[f"{cfg.env_id}-{cfg.dmc_task}", f"seed={str(cfg.seed)}"],
            # sync_tensorboard=True,
            config=cfg_dict,
            name=cfg.run_name,
            monitor_gym=cfg.monitor_gym,
            save_code=True,
            dir=os.path.join(get_original_cwd(), "output"),
        )
        wandb.run.summary["timeout_min"] = HydraConfig.get().launcher.timeout_min
    pprint.pprint(cfg_dict)
    pprint.pprint(HydraConfig.get().launcher)

    ###### Prepare replay buffer ######
    rb = ReplayBuffer(
        int(cfg.buffer_size),
        envs.single_observation_space,
        envs.single_action_space,
        # "auto",
        torch.device(cfg.device),
        nstep=cfg.agent.nstep,
        #        handle_timeout_termination=False,
        discount=cfg.agent.discount,
        train_validation_split=cfg.train_validation_split,
    )

    ###### Init agent ######
    agent = hydra.utils.instantiate(
        cfg.agent,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
    )

    ###### Convert episode stuff to global steps ######
    total_timesteps = int(cfg.num_episodes * cfg.max_episode_steps / cfg.action_repeat)
    eval_every_steps = int(
        cfg.eval_every_episodes * cfg.max_episode_steps / cfg.action_repeat
    )
    learning_starts = int(
        cfg.random_episodes * cfg.max_episode_steps / cfg.action_repeat
    )

    ###########################
    ###### Training loop ######
    ###########################
    episode_idx, start_time = 0, time.time()
    obs, _ = envs.reset(seed=cfg.seed)

    for global_step in range(total_timesteps):
        ###### Select action ######
        if global_step < learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            # TODO set t0 properly
            actions = agent.select_action(observation=obs, eval_mode=False, t0=False)

        ###### Execute action in environments ######
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

        ###### Record training metrics ######
        if "final_info" in infos:
            for info in infos["final_info"]:
                logger.info(
                    f"Episode: {episode_idx} | Global step: {global_step} | Return: {info['episode']['r']}"
                )
                episode_length = info["episode"]["l"]
                episode_idx += 1

                if cfg.use_wandb:
                    wandb.log(
                        {
                            "episodic_return": info["episode"]["r"],
                            "episodic_length": info["episode"]["l"],
                            # "global_step": global_step,
                            "env_step": global_step * cfg.action_repeat,
                            # "episode": episode_idx,
                        }
                    )
                break

        ###### Save data to reply buffer; handle `terminal_observation` ######
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncateds):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminateds, truncateds, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        ###### Training ######
        if global_step > learning_starts:
            if "final_info" in infos:  # Update after every episode
                if isinstance(episode_length, np.ndarray):
                    episode_length = episode_length[0]
                # num_updates = cfg.utd_ratio * episode_length  # TODO inc. frame skip
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
                            # "SPS": int(global_step / (time.time() - start_time)),
                            "num_new_transitions": num_new_transitions,
                            # "action_repeat": cfg.action_repeat,
                            "env_step": global_step * cfg.action_repeat,
                            # "episode": episode_idx,
                            "elapsed_time": time.time() - start_time,
                        }
                    )
                    wandb.log({"train/": train_metrics})

            ###### Evaluation ######
            if global_step % eval_every_steps == 0:
                mean_reward, std_reward = evaluate_policy(
                    agent, eval_envs, n_eval_episodes=cfg.num_eval_episodes
                )
                logger.info(f"reward mean {mean_reward} std: {mean_reward}")
                eval_metrics = {
                    "episodic_return": mean_reward,
                    "episodic_return_std": std_reward,
                    # "mean_reward": mean_reward,
                    # "std_reward": std_reward,
                    # "global_step": global_step,
                    "env_step": global_step * cfg.action_repeat,
                    "episode": episode_idx,
                    "elapsed_time": time.time() - start_time,
                }
                if cfg.use_wandb:
                    wandb.log({"eval/": eval_metrics})

    envs.close()


if __name__ == "__main__":
    train()  # pyright: ignore
