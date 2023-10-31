#!/usr/bin/env python3
import logging
import pprint
import random
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gymnasium as gym
import hydra
import omegaconf
from configs.base import TrainConfig


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
# @hydra.main(version_base=None, config_name="train_config")
def train(cfg: TrainConfig):
    import numpy as np
    import torch
    from hydra.utils import get_original_cwd
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.evaluation import evaluate_policy
    from utils.env import make_env

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
                env_id=cfg.env.env_id,
                seed=cfg.seed,
                idx=0,
                capture_video=cfg.capture_train_video,
                run_name=cfg.run_name,
                max_episode_steps=cfg.max_episode_steps,
                action_repeat=cfg.action_repeat,
                dmc_task=cfg.env.dmc_task,
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
                env_id=cfg.env.env_id,
                seed=cfg.seed + 100,
                idx=0,
                capture_video=cfg.capture_train_video,
                run_name=cfg.run_name,
                max_episode_steps=cfg.max_episode_steps,
                action_repeat=cfg.action_repeat,
                dmc_task=cfg.env.dmc_task,
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
