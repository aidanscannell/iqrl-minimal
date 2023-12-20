#!/usr/bin/env python3
import logging
from typing import Any, List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import agents
import gymnasium as gym
import helper as h
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from custom_types import Action, Agent, EvalMode, T0
from gymnasium.spaces import Box, Space
from helper import soft_update_params
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class Encoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        act_fn=nn.ELU,
        normalize: bool = True,
        simplex_dim: int = 10,
    ):
        super().__init__()
        in_dim = np.array(observation_space.shape).prod()
        # TODO data should be normalized???
        self.latent_dim = latent_dim
        self._mlp = h.mlp(
            in_dim=in_dim, mlp_dims=mlp_dims, out_dim=latent_dim, act_fn=act_fn
        )
        self.normalize = normalize
        self.simplex_dim = simplex_dim
        self.reset(full_reset=True)

    def forward(self, x):
        z = self._mlp(x)
        # breakpoint()
        if self.normalize:
            z = h.simnorm(z, V=self.simplex_dim)
        # print(f"min z {torch.min(z)}")
        # print(f"max z {torch.max(z)}")
        return z

    def reset(self, full_reset: bool = False):
        if full_reset:
            self.apply(h.orthogonal_init)
        else:
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])


class Decoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        act_fn=nn.ELU,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        out_dim = np.array(observation_space.shape).prod()
        self._mlp = h.mlp(
            in_dim=latent_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=act_fn
        )
        self.reset(full_reset=True)

    def forward(self, z):
        x = self._mlp(z)
        return x

    def reset(self, full_reset: bool = False):
        if full_reset:
            self.apply(h.orthogonal_init)
        else:
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])


class AE(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        act_fn=nn.ELU,
        normalize: bool = True,
        simplex_dim: int = 10,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.mlp_dims = mlp_dims
        self.latent_dim = latent_dim
        self.act_fn = act_fn
        self.encoder = Encoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
            normalize=normalize,
            simplex_dim=simplex_dim,
        )
        self.decoder = Decoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

    # def reset(self):
    #     self.encoder.reset()
    #     self.decoder.reset()
    def reset(self, full_reset: bool = False):
        logger.info("Resetting encoder/decoder params")
        self.encoder.reset(full_reset=full_reset)
        self.decoder.reset(full_reset=full_reset)


class AEDDPG(Agent):
    def __init__(
        self,
        # DDPG config
        observation_space: Space,
        action_space: Box,
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
        ae_normalize: bool = True,
        simplex_dim: int = 10,
        # encoder_reset_params_freq: int = 10000,  # reset enc params after X param updates
        device: str = "cuda",
        name: str = "AEDDPG",
    ):
        # super().__init__(
        #     observation_space=observation_space, action_space=action_space, name=name
        # )
        self.ae_learning_rate = ae_learning_rate
        self.ae_batch_size = ae_batch_size
        self.ae_utd_ratio = ae_utd_ratio
        # self.ae_num_updates = ae_num_updates
        self.latent_dim = latent_dim

        self.ae_tau = ae_tau

        self.device = device

        self.ae = AE(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
            normalize=ae_normalize,
            simplex_dim=simplex_dim,
        ).to(device)
        self.ae_target = AE(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
            normalize=ae_normalize,
            simplex_dim=simplex_dim,
        ).to(device)
        self.ae_opt = torch.optim.AdamW(self.ae.parameters(), lr=ae_learning_rate)
        self.ae_target.load_state_dict(self.ae.state_dict())

        self.ae_patience = ae_patience
        self.ae_min_delta = ae_min_delta
        self.ae_early_stopper = h.EarlyStopper(
            patience=ae_patience, min_delta=ae_min_delta
        )

        # TODO make a space for latent states
        # latent_observation_space = observation_space
        # high = np.array(levels).prod()
        # TODO is this the right way to make observation space??
        # TODO Should we bound z in -100,100 instead of -inf,inf??
        if ae_normalize:
            self.latent_observation_space = gym.spaces.Box(
                low=0, high=1, shape=(latent_dim,)
            )
        else:
            self.latent_observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(latent_dim,)
                # low=0.0, high=high, shape=(latent_dim,)
            )
        print(f"latent_observation_space {self.latent_observation_space}")
        # Init ddpg agent
        self.ddpg = agents.DDPG(
            observation_space=self.latent_observation_space,
            action_space=action_space,
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
            name=name,
        )

        # self.use_memory = False
        # self.use_memory_flag = False

        self.encoder_update_conter = 0
        # self.encoder_reset_params_freq = encoder_reset_params_freq
        self.reset_params_freq = reset_params_freq

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        method = "same"
        if method == "same":
            return self.update_1(
                replay_buffer=replay_buffer, num_new_transitions=num_new_transitions
            )
        else:
            return self.update_2(
                replay_buffer=replay_buffer, num_new_transitions=num_new_transitions
            )

    def update_1(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and DDPG at same time"""
        num_updates = num_new_transitions * self.ddpg.utd_ratio
        info = {}

        logger.info(f"Performing {num_updates} AEDDPG updates...")
        for i in range(num_updates):
            batch = replay_buffer.sample(self.ddpg.batch_size)
            info.update(self.update_representation_step(batch=batch))

            # Map observations to latent
            latent_obs = self.ae_target.encoder(batch.observations)
            latent_next_obs = self.ae_target.encoder(batch.next_observations)
            # batch.observation = latent_obs.to(torch.float).detach()
            # batch.next_observation = latent_next_obs.to(torch.float).detach()
            batch = ReplayBufferSamples(
                observations=latent_obs.to(torch.float).detach(),
                actions=batch.actions,
                next_observations=latent_next_obs.to(torch.float).detach(),
                dones=batch.dones,
                rewards=batch.rewards,
            )

            # DDPG on latent representation
            info.update(self.ddpg.update_step(batch=batch))

            # Reset ae/actor/critic after a fixed number of parameter updates
            if self.ddpg.critic_update_counter % self.ddpg.reset_params_freq == 0:
                self.reset(full_reset=False)

            if i % 100 == 0:
                logger.info(f"Iteration {i} rec_loss {info['rec_loss']}")
                if wandb.run is not None:
                    wandb.log(info)

        logger.info("Finished training AEDDPG")

        return info

    def update_2(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and then do DDPG"""
        num_updates = num_new_transitions * self.ddpg.utd_ratio
        # num_updates = num_new_transitions * self.utd_ratio
        info = {}

        logger.info("Training AE...")
        for i in range(num_updates):
            batch = replay_buffer.sample(self.ae_batch_size)
            info.update(self.update_representation_step(batch=batch))

            if i % 100 == 0:
                logger.info(f"Iteration {i} rec_loss {info['rec_loss']}")
                if wandb.run is not None:
                    wandb.log(info)
        logger.info("Finished training AE")

        logger.info(f"Performing {num_updates} DDPG updates...")
        for i in range(num_updates):
            batch = replay_buffer.sample(self.ddpg.batch_size)

            # Map observations to latent
            latent_obs = self.ae_target.encoder(batch.observations)
            latent_next_obs = self.ae_target.encoder(batch.next_observations)
            # batch.observation = latent_obs.to(torch.float).detach()
            # batch.next_observation = latent_next_obs.to(torch.float).detach()
            batch = ReplayBufferSamples(
                observations=latent_obs.to(torch.float).detach(),
                actions=batch.actions,
                next_observations=latent_next_obs.to(torch.float).detach(),
                dones=batch.dones,
                rewards=batch.rewards,
            )

            # DDPG on latent representation
            info.update(self.ddpg.update_step(batch=batch))

            # Reset actor/critic after a fixed number of parameter updates
            if self.critic_update_counter % self.reset_params_freq == 0:
                self.reset(full_reset=False)

            if i % 100 == 0:
                if wandb.run is not None:
                    wandb.log(info)

        logger.info("Finished training DDPG")

        return info

    def update_representation_step(self, batch: ReplayBufferSamples):
        # Reset encoder after a fixed number of updates
        self.encoder_update_conter += 1
        # if self.encoder_update_conter % self.encoder_reset_params_freq == 0:
        #     logger.info("Resetting AE params")
        #     self.ae.reset()
        #     self.ae_target.load_state_dict(self.ae.state_dict())
        #     self.ae_opt = torch.optim.AdamW(
        #         self.ae.parameters(), lr=self.ae_learning_rate
        #     )

        # x_train = torch.concat([batch.observations, batch.next_observations], 0)
        x_train = batch.observations

        x_rec, z = self.ae(x_train)
        rec_loss = (x_rec - x_train).abs().mean()

        loss = rec_loss
        self.ae_opt.zero_grad()
        loss.backward()
        self.ae_opt.step()

        # Update the target network
        soft_update_params(self.ae, self.ae_target, tau=self.ae_tau)

        info = {
            "rec_loss": rec_loss.item(),
            "loss": loss.item(),
        }
        return info

    # def update_representation(
    #     self, replay_buffer: ReplayBuffer, num_new_transitions: int
    # ):
    #     num_updates = num_new_transitions * self.ae_utd_ratio
    #     ae_early_stopper = EarlyStopper(
    #         patience=self.ae_patience, min_delta=self.ae_min_delta
    #     )
    #     info = {}
    #     i = 0
    #     for _ in range(num_updates):
    #         # Reset encoder after a fixed number of updates
    #         self.encoder_update_conter += 1
    #         if self.encoder_update_conter % self.encoder_reset_params_freq == 0:
    #             logger.info("Resetting AE params")
    #             self.ae.reset()
    #             self.ae_target.load_state_dict(self.ae.state_dict())
    #             self.ae_opt = torch.optim.AdamW(
    #                 self.ae.parameters(), lr=self.ae_learning_rate
    #             )

    #         batch = replay_buffer.sample(self.ae_batch_size)
    #         x_train = torch.concat([batch.observations, batch.next_observations], 0)

    #         x_rec, z = self.ae(x_train)
    #         rec_loss = (x_rec - x_train).abs().mean()

    #         loss = rec_loss
    #         self.ae_opt.zero_grad()
    #         loss.backward()
    #         self.ae_opt.step()

    #         # Update the target network
    #         for param, target_param in zip(
    #             self.ae.parameters(), self.ae_target.parameters()
    #         ):
    #             target_param.data.copy_(
    #                 self.ae_tau * param.data + (1 - self.ae_tau) * target_param.data
    #             )

    #         if i % 100 == 0:
    #             logger.info(f"Iteration {i} rec_loss {rec_loss}")
    #             if wandb.run is not None:
    #                 wandb.log(
    #                     {
    #                         "rec_loss": rec_loss.item(),
    #                         "loss": loss.item(),
    #                     }
    #                 )

    #         i += 1
    #         # if ae_early_stopper(rec_loss):
    #         #     logger.info("Early stopping criteria met, stopping AE training...")
    #         #     break

    #     return info

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        observation = torch.Tensor(observation).to(self.device)
        _, z = self.ae(observation)
        z = z.to(torch.float)
        return self.ddpg.select_action(observation=z, eval_mode=eval_mode, t0=t0)

    def reset(self, full_reset: bool = False):
        logger.info("Restting agent...")
        self.ae.reset(full_reset=full_reset)
        # self.ae_target.reset(full_reset=full_reset)
        self.ae_target.load_state_dict(self.ae.state_dict())
        self.ae_opt = torch.optim.AdamW(
            list(self.ae.parameters()), lr=self.ae_learning_rate
        )

        self.ddpg.reset(full_reset=full_reset)
