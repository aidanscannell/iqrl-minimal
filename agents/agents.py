#!/usr/bin/env python3
import abc
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from agents.actors import Actor, MLPActor
from agents.critics import Critic, MLPCritic
from agents.encoders import AE, Encoder, FSQAutoEncoder
from custom_types import Action, EvalMode, T0
from gymnasium.spaces import Box, Space
from helper import EarlyStopper, soft_update_params
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


@dataclass
class Agent(abc.ABC):
    observation_space: Space
    action_space: Box
    name: str = "BaseAgent"

    @abc.abstractmethod
    def select_action(
        self, observation, eval_mode: EvalMode = False, t0: T0 = None
    ) -> Action:
        raise NotImplementedError

    @abc.abstractmethod
    def update(
        self, replay_buffer: ReplayBuffer, num_new_transitions: int
    ) -> Optional[dict]:
        raise NotImplementedError

    def predict(
        self, observation, state=None, episode_start=None, deterministic: bool = False
    ):
        """Allows agent to work with Stablebaselines evaluate_policy"""
        action = self.select_action(observation=observation, eval_mode=True)
        # action = self.select_action(observation=observation, eval_mode=True, t0=t0)
        recurrent_state = None
        return action, recurrent_state


class DDPG(Agent):
    def __init__(
        self,
        observation_space: Space,
        action_space: Box,
        mlp_dims: List[int] = [512, 512],
        act_fn=nn.ELU,
        exploration_noise: float = 0.2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        utd_ratio: int = 1,  # parameter update-to-data ratio
        actor_update_freq: int = 1,  # update actor less frequently than critic
        reset_params_freq: int = 100000,  # reset params after this many param updates
        # nstep: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
        name: str = "DDPG",
    ):
        super().__init__(
            observation_space=observation_space, action_space=action_space, name=name
        )
        self.mlp_dims = mlp_dims
        self.act_fn = act_fn
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.actor_update_freq = actor_update_freq  # Should be 1 for true DDPG
        self.reset_params_freq = reset_params_freq
        # self.nstep = nstep
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Init actor and it's target
        self.actor = MLPActor(
            observation_space=observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
            act_fn=act_fn,
        ).to(device)
        self.target_actor = MLPActor(
            observation_space=observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
            act_fn=act_fn,
        ).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Init critic and it's targets
        self.critic = MLPCritic(
            observation_space, action_space, mlp_dims=mlp_dims, act_fn=act_fn
        ).to(device)
        self.target_critic = MLPCritic(
            observation_space, action_space, mlp_dims=mlp_dims, act_fn=act_fn
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            list(self.critic.parameters()), lr=learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()), lr=learning_rate
        )

        # Counters for number of param updates
        self.critic_update_counter = 0
        self.actor_update_counter = 0

        self.reset_flag = False

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        self.reset_flag = False
        num_updates = num_new_transitions * self.utd_ratio
        logger.info(f"Performing {num_updates} DDPG updates")

        for i in range(num_updates):
            batch = replay_buffer.sample(self.batch_size)

            info = self.update_step(batch=batch, i=i)

            # Reset actor/critic after a fixed number of parameter updates
            if self.critic_update_counter % self.reset_params_freq == 0:
                self.reset(full_reset=False)

        return info

    def update_step(self, batch, i: int = -1) -> dict:
        info = {}

        # Update critic
        info.update(self.critic_update_step(data=batch))

        # Update actor less frequently than critic
        if self.critic_update_counter % self.actor_update_freq == 0:
            info.update(self.actor_update_step(data=batch))

        if i % 100 == 0:
            if wandb.run is not None:
                wandb.log(info)

        return info

    def critic_update_step(self, data) -> dict:
        # Reset critic after a fixed number of parameter updates
        self.critic_update_counter += 1
        # if self.critic_update_counter % self.reset_params_freq == 0:
        #     logger.info("Resetting critic's params")
        #     self.critic.reset()
        #     self.target_critic.load_state_dict(self.critic.state_dict())
        #     self.critic_opt = torch.optim.AdamW(
        #         self.critic.parameters(), lr=self.learning_rate
        #     )

        with torch.no_grad():
            clipped_noise = (
                torch.randn_like(data.actions, device=self.device) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip) * self.target_actor.action_scale

            next_state_actions = (
                self.target_actor(data.next_observations) + clipped_noise
            ).clamp(self.action_space.low[0], self.action_space.high[0])
            critic_next_target = self.target_critic(
                data.next_observations, next_state_actions
            )
            next_q_value = data.rewards.flatten() + (
                1 - data.dones.flatten()
            ) * self.gamma * (critic_next_target).view(-1)

        critic_a_values = self.critic(data.observations, data.actions).view(-1)
        critic_loss = F.mse_loss(critic_a_values, next_q_value)

        # Optimize the model
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # Update the target network
        # with torch.no_grad():
        #     for param, target_param in zip(
        #         self.critic.parameters(), self.target_critic.parameters()
        #     ):
        #         target_param.data.copy_(
        #             self.tau * param.data + (1 - self.tau) * target_param.data
        #         )
        soft_update_params(self.critic, self.target_critic, tau=self.tau)

        info = {
            "critic_values": critic_a_values.mean().item(),
            "critic_loss": critic_loss.item(),
            "critic_update_counter": self.critic_update_counter,
        }
        return info

    def actor_update_step(self, data) -> dict:
        self.actor_update_counter += 1
        actor_loss = -self.critic(
            data.observations, self.actor(data.observations)
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target network
        soft_update_params(self.actor, self.target_actor, tau=self.tau)
        # with torch.no_grad():
        #     for param, target_param in zip(
        #         self.actor.parameters(), self.target_actor.parameters()
        #     ):
        #         target_param.data.copy_(
        #             self.tau * param.data + (1 - self.tau) * target_param.data
        #         )

        info = {
            "actor_loss": actor_loss.item(),
            "actor_update_counter": self.actor_update_counter,
        }
        return info

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        actions = self.actor(torch.Tensor(observation).to(self.device))
        if not eval_mode:
            actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
        actions = actions.cpu().numpy()
        actions = actions.clip(self.action_space.low, self.action_space.high)
        return actions

    def reset(self, full_reset: bool = False):
        logger.info("Resetting actor/critic params")
        self.actor.reset(full_reset=full_reset)
        self.critic.reset(full_reset=full_reset)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=self.learning_rate
        )
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.reset_flag = True
        # TODO more updates after resetting?
        # for j in range(replay_buffer.size() - num_updates):
        #     batch = replay_buffer.sample(self.batch_size)
        #     info = self.update_step(batch=batch, i=i + j)
        # self.critic_update_counter = 1
        # self.actor_update_counter = 1
        # break

        # self.critic_update_counter += 1


class ActorCritic(Agent, abc.ABC):
    observation_space: Space
    action_space: Box
    actor: Actor
    critic: Critic
    reset_flag: bool = False  # True if params have been reset
    name: str = "BaseActorCriticAgent"

    def reset(self, full_reset: bool = False):
        raise NotImplementedError


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
        if self.actor_critic.reset_flag:
            self.encoder.reset(full_reset=False)
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
        # breakpoint()
        z = z.to(torch.float)
        return self.actor_critic.select_action(
            observation=z, eval_mode=eval_mode, t0=t0
        )

    def reset(self, full_reset: bool = False):
        logger.info("Restting agent...")
        self.encoder.reset(full_reset=full_reset)
        self.actor_critic.reset(full_reset=full_reset)


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
        reset_params_freq: int = 40000,  # reset params after this many critic param updates
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
        # encoder_reset_params_freq: int = 10000,  # reset enc params after X param updates
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
            # encoder_reset_params_freq=encoder_reset_params_freq,
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


class VQDDPG(LatentActorCritic):
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
        reset_params_freq: int = 40000,  # reset params after this many critic param updates
        # nstep: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        # VQ config
        ae_learning_rate: float = 3e-4,
        ae_batch_size: int = 128,
        # ae_num_updates: int = 1000,
        ae_utd_ratio: int = 1,
        ae_patience: int = 100,
        ae_min_delta: float = 0.0,
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        ae_tau: float = 0.005,
        # encoder_reset_params_freq: int = 10000,  # reset enc params after X param updates
        device: str = "cuda",
        name: str = "VQDDPG",
    ):
        build_vq_fn = partial(
            FSQAutoEncoder,
            mlp_dims=mlp_dims,
            levels=levels,
            act_fn=act_fn,
            learning_rate=ae_learning_rate,
            batch_size=ae_batch_size,
            utd_ratio=ae_utd_ratio,
            tau=ae_tau,
            # encoder_reset_params_freq=encoder_reset_params_freq,
            early_stopper=EarlyStopper(patience=ae_patience, min_delta=ae_min_delta),
            device=device,
            name="VQ",
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
            build_encoder_fn=build_vq_fn,
            build_actor_critic_fn=build_actor_critic_fn,
        )
        self.device = device
        self.name = name
