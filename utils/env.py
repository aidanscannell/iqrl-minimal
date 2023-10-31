#!/usr/bin/env python3
import logging
from typing import List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gymnasium as gym

from .dmc2gymnasium import DMCGym


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
