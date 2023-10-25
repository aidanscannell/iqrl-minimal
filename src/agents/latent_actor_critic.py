#!/usr/bin/env python3
import logging
from functools import partial
from typing import Callable, List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Space
from src.agents import DDPG
from src.agents.base import ActorCritic
from src.agents.utils import EarlyStopper
from src.custom_types import EvalMode, T0
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from .base import Agent
from .encoders import Encoder, AE


class LatentActorCritic(Agent):
    def __init__(
        self,
        observation_space: Space,
        action_space: Box,
        build_encoder_fn: Callable[[Space], Encoder] = partial(AE),
        build_actor_critic_fn: Callable[[Space, Box], ActorCritic] = partial(DDPG),
        device: str = "cuda",
        name: str = "LatentActorCritic",
    ):
        super().__init__(
            observation_space=observation_space, action_space=action_space, name=name
        )
        self.encoder = build_encoder_fn(observation_space)
        self.device = device

        # TODO make a space for latent states
        # latent_observation_space = observation_space
        # high = np.array(levels).prod()
        # TODO is this the right way to make observation space??
        # TODO Should we bound z in -100,100 instead of -inf,inf??
        # Init actor critic agent with latent observation space
        self.latent_observation_space = gym.spaces.Box(
            # low=self.encoder.latent_low,
            # high=self.encoder.latent_high,
            low=-np.inf,
            high=np.inf,
            shape=(self.encoder.latent_dim,)
            # low=0.0, high=high, shape=(latent_dim,)
        )
        print(f"latent_observation_space {self.latent_observation_space}")
        self.actor_critic = build_actor_critic_fn(
            self.latent_observation_space, action_space
        )

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        logger.info("Training representation...")
        info = self.encoder.update(
            replay_buffer=replay_buffer, num_new_transitions=num_new_transitions
        )
        logger.info("Finished training representation")

        encoder = self.encoder

        class LatentReplayBuffer:
            def sample(self, batch_size: int):
                batch = replay_buffer.sample(batch_size=batch_size)
                latent_obs = encoder(batch.observations, target=True)
                latent_next_obs = encoder(batch.next_observations, target=True)
                batch = ReplayBufferSamples(
                    observations=latent_obs.to(torch.float).detach(),
                    actions=batch.actions,
                    next_observations=latent_next_obs.to(torch.float).detach(),
                    dones=batch.dones,
                    rewards=batch.rewards,
                )
                # breakpoint()
                return batch

        latent_replay_buffer = LatentReplayBuffer()
        logger.info("Training actor critic...")
        info.update(
            self.actor_critic.update(
                replay_buffer=latent_replay_buffer,
                num_new_transitions=num_new_transitions,
            )
        )
        logger.info("Finished training actor critic")
        return info

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        observation = torch.Tensor(observation).to(self.device)
        z = self.encoder(observation, target=False)
        z = z.to(torch.float)
        return self.actor_critic.select_action(
            observation=z, eval_mode=eval_mode, t0=t0
        )


class AEDDPG(LatentActorCritic):
    def __init__(
        self,
        observation_space: Space,
        action_space: Box,
        # DDPG config
        mlp_dims: List[int] = [512, 512],
        act_fn=nn.ELU,
        exploration_noise: float = 0.2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        utd_ratio: int = 1,  # DDPG parameter update-to-data ratio
        actor_update_freq: int = 1,  # update actor less frequently than critic
        reset_params_freq: int = 100000,  # reset params after this many param updates
        # nstep: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        # AE config
        ae_learning_rate: float = 3e-4,
        ae_batch_size: int = 128,
        # ae_num_updates: int = 1000,
        ae_utd_ratio: int = 1,
        ae_patience: int = 100,
        ae_min_delta: float = 0.0,
        latent_dim: int = 20,
        ae_tau: float = 0.005,
        encoder_reset_params_freq: int = 10000,  # reset enc params after X param updates
        device: str = "cuda",
        name: str = "AEDDPG",
    ):
        build_ae_fn = partial(
            AE,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
            learning_rate=ae_learning_rate,
            batch_size=ae_batch_size,
            utd_ratio=ae_utd_ratio,
            tau=ae_tau,
            encoder_reset_params_freq=encoder_reset_params_freq,
            early_stopper=EarlyStopper(patience=ae_patience, min_delta=ae_min_delta),
            device=device,
            name="AE",
        )
        build_actor_critic_fn = partial(
            DDPG,
            mlp_dims=mlp_dims,
            act_fn=act_fn,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            learning_rate=learning_rate,
            batch_size=batch_size,
            utd_ratio=utd_ratio,
            actor_update_freq=actor_update_freq,
            reset_params_freq=reset_params_freq,
            gamma=gamma,
            tau=tau,
            device=device,
            name="DDPG",
        )
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            build_encoder_fn=build_ae_fn,
            build_actor_critic_fn=build_actor_critic_fn,
        )
        self.device = device
        self.name = name
