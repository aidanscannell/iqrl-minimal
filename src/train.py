#!/usr/bin/env python3
import logging
import os
import random
import shutil
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import src
import torch
import wandb
from dm_env import StepType
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from src.utils import ReplayBuffer, set_seed_everywhere


# torch.set_default_dtype(torch.float64)


@hydra.main(version_base="1.3", config_path="./configs", config_name="main")
def train(cfg: DictConfig):
    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = "cpu"
    print("Using device: {}".format(cfg.device))
    cfg.episode_length = cfg.episode_length // cfg.env.action_repeat
    num_train_steps = cfg.num_train_episodes * cfg.episode_length

    env = hydra.utils.instantiate(cfg.env)
    print("ENV {}".format(env.observation_spec()))
    ts = env.reset()
    print(ts["observation"])
    eval_env = hydra.utils.instantiate(cfg.env, seed=cfg.env.seed + 42)

    cfg.obs_shape = tuple(int(x) for x in env.observation_spec().shape)
    # cfg.state_dim = cfg.state_dim[0]
    cfg.action_shape = tuple(int(x) for x in env.action_spec().shape)
    cfg.action_dim = cfg.action_shape[0]

    ###### Set up workspace ######
    # work_dir = (
    #     Path().cwd()
    #     / "logs"
    #     / cfg.alg_name
    #     # / cfg.name
    #     / cfg.env.env_name
    #     / cfg.env.task_name
    #     / str(cfg.random_seed)
    # )
    if cfg.wandb.use_wandb:  # Initialise WandB
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            # monitor_gym=True,
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
        try:
            shutil.copytree(
                os.path.abspath(".hydra"),
                os.path.join(os.path.join(get_original_cwd(), wandb.run.dir), "hydra"),
            )
            wandb.save("hydra")
        except FileExistsError:
            pass

    print("Making recorder")
    video_recorder = src.utils.VideoRecorder(run.dir) if cfg.save_video else None
    print("Made recorder")

    # Create replay buffer
    print("Making replay buffer")
    print("num_train_steps {}".format(num_train_steps))
    # replay_memory = ReplayBuffer(
    #     capacity=num_train_steps, batch_size=cfg.batch_size, device=cfg.device
    # )
    replay_memory = hydra.utils.instantiate(cfg.replay_buffer)
    print("Made replay buffer")

    agent = hydra.utils.instantiate(cfg.agent)
    print("Made agent")

    start_time = time.time()
    last_time = start_time
    global_step = 0
    for episode_idx in range(cfg.num_train_episodes):
        logger.info("Episode {} | Collecting data".format(episode_idx))

        # Collect trajectory
        time_step = env.reset()

        if cfg.save_video:
            video_recorder.init(env, enabled=True)

        episode_return = 0
        t = 0
        while not time_step.last():
            # logger.info("Timestep: {}".format(t))
            if episode_idx <= cfg.init_random_episodes:
                action = np.random.uniform(-1, 1, env.action_spec().shape).astype(
                    dtype=np.float64
                    # dtype=env.action_spec().dtype
                )
            else:
                action_select_time = time.time()
                action = agent.select_action(
                    time_step.observation,
                    eval_mode=False,
                    t0=time_step.step_type == StepType.FIRST,
                )

                action_select_end_time = time.time()
                if t % 100 == 0:
                    logger.info(
                        "timestep={} took {}s to select action".format(
                            t, action_select_end_time - action_select_time
                        )
                    )
                action = action.cpu().numpy()

            ##### Step the environment #####
            time_step = env.step(action)

            replay_memory.push(
                state=torch.Tensor(time_step["observation"]).to(cfg.device),
                action=torch.Tensor(time_step["action"]).to(cfg.device),
                next_state=torch.Tensor(time_step["observation"]).to(cfg.device),
                reward=torch.Tensor([time_step["reward"]]).to(cfg.device),
            )

            global_step += 1
            episode_return += time_step["reward"]
            t += 1
            if cfg.save_video:
                video_recorder.record(env)

        logger.info("Finished collecting {} time steps".format(t))

        # Log training metrics
        env_step = global_step * cfg.env.action_repeat
        elapsed_time = time.time() - last_time
        total_time = time.time() - start_time
        last_time = time.time()
        train_metrics = {
            "episode": episode_idx,
            "step": global_step,
            "env_step": env_step,
            "episode_time": elapsed_time,
            "total_time": total_time,
            "episode_return": episode_return,
        }
        logger.info(
            "TRAINING | Episode: {} | Reward: {}".format(episode_idx, episode_return)
        )
        if cfg.wandb.use_wandb:
            wandb.log({"train/": train_metrics})

        ##### Train agent #####
        # for _ in range(cfg.episode_length // cfg.update_every_steps):
        if episode_idx >= cfg.init_random_episodes:
            logger.info("Training agent...")
            agent.train(replay_memory)
            logger.info("Finished agent")

            if cfg.save_video:
                # G = src.utils.evaluate(
                #     eval_env,
                #     agent,
                #     episode_idx=episode_idx,
                #     num_episodes=1,
                #     online_updates=False,
                #     online_update_freq=cfg.online_update_freq,
                #     video=video_recorder,
                #     device=cfg.device,
                # )
                print("saving video episode: {}".format(episode_idx))
                video_recorder.save(episode_idx)

            # # Log rewards/videos in eval env
            # if episode_idx % cfg.eval_episode_freq == 0:
            #     # print("Evaluating {}".format(episode_idx))
            #     logger.info("Starting eval episodes")
            #     G_no_online_updates = src.utils.evaluate(
            #         eval_env,
            #         agent,
            #         episode_idx=episode_idx,
            #         num_episodes=1,
            #         online_updates=False,
            #         online_update_freq=cfg.online_update_freq,
            #         video=video_recorder,
            #         device=cfg.device,
            #     )

            #     # Gs = utils.evaluate(
            #     #     eval_env,
            #     #     agent,
            #     #     episode_idx=episode_idx,
            #     #     # num_episode=cfg.eval_episode_freq,
            #     #     num_episodes=1,
            #     #     # num_episodes=10,
            #     #     # video=video_recorder,
            #     # )
            #     # print("DONE EVALUATING")
            #     # eval_episode_reward = np.mean(Gs)
            #     env_step = global_step * cfg.env.action_repeat
            #     eval_metrics = {
            #         "episode": episode_idx,
            #         "step": global_step,
            #         "env_step": env_step,
            #         "episode_time": elapsed_time,
            #         "total_time": total_time,
            #         # "episode_reward": eval_episode_reward,
            #         "episode_return/no_online_updates": G_no_online_updates,
            #     }
            #     logger.info(
            #         "EVAL (no updates) | Episode: {} | Retrun: {}".format(
            #             episode_idx, G_no_online_updates
            #         )
            #     )

            #     if cfg.wandb.use_wandb:
            #         wandb.log({"eval/": eval_metrics})


if __name__ == "__main__":
    train()  # pyright: ignore
