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
from helper import SimNorm, soft_update_params
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class Encoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        normalize: bool = True,
        simplex_dim: int = 10,
    ):
        super().__init__()
        in_dim = np.array(observation_space.shape).prod()
        # TODO data should be normalized???
        if normalize:
            act_fn = SimNorm(dim=simplex_dim)
        else:
            act_fn = None
        self.latent_dim = latent_dim
        self._mlp = h.mlp(
            in_dim=in_dim,
            mlp_dims=mlp_dims,
            out_dim=latent_dim,
            act_fn=act_fn,
        )
        self.normalize = normalize
        self.simplex_dim = simplex_dim
        self.reset(reset_type="full")

    def forward(self, x):
        z = self._mlp(x)
        # breakpoint()
        # if self.normalize:
        # z = h.simnorm(z, V=self.simplex_dim)
        # print(f"min z {torch.min(z)}")
        # print(f"max z {torch.max(z)}")
        return z

    def reset(self, reset_type: str = "last-layer"):
        if reset_type in "full":
            h.orthogonal_init(self.parameters())
        elif reset_type in "last-layer":
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])
        else:
            raise NotImplementedError


class Decoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        out_dim = np.array(observation_space.shape).prod()
        self._mlp = h.mlp(in_dim=latent_dim, mlp_dims=mlp_dims, out_dim=out_dim)
        self.reset(reset_type="full")

    def forward(self, z):
        x = self._mlp(z)
        return x

    def reset(self, reset_type: str = "last-layer"):
        if reset_type in "full":
            h.orthogonal_init(self.parameters())
            # self.apply(h.orthogonal_init)
        elif reset_type in "last-layer":
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])
        else:
            raise NotImplementedError


class MLPDynamics(nn.Module):
    def __init__(
        self,
        action_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        normalize: bool = True,
        simplex_dim: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = np.array(action_space.shape).prod() + latent_dim
        if normalize:
            act_fn = SimNorm(dim=simplex_dim)
        else:
            act_fn = None
        # TODO data should be normalized???
        self._mlp = h.mlp(
            in_dim=in_dim,
            mlp_dims=mlp_dims,
            out_dim=latent_dim,
            # out_dim=latent_dim + 1,
            act_fn=act_fn,
        )
        self.reset(reset_type="full")

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        z = self._mlp(x)
        return z
        # r = z[..., -1:]
        # return z[..., :-1], r

    def reset(self, reset_type: str = "last-layer"):
        if reset_type in "full":
            h.orthogonal_init(self.parameters())
            # self.apply(h.orthogonal_init)
        elif reset_type in "last-layer":
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])
        else:
            raise NotImplementedError


class MLPReward(nn.Module):
    def __init__(
        self,
        action_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        # normalize: bool = True,
        # simplex_dim: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = np.array(action_space.shape).prod() + latent_dim
        # if normalize:
        #     act_fn = SimNorm(dim=simplex_dim)
        # else:
        # act_fn = None
        # TODO data should be normalized???
        self._mlp = h.mlp(
            in_dim=in_dim,
            mlp_dims=mlp_dims,
            out_dim=1,
            act_fn=None,  # This will use Mish
        )
        self.reset(reset_type="full")

    def forward(self, z, a):
        x = torch.cat([z, a], 1)
        r = self._mlp(x)
        return r

    def reset(self, reset_type: str = "last-layer"):
        if reset_type in "full":
            h.orthogonal_init(self.parameters())
            # self.apply(h.orthogonal_init)
        elif reset_type in "last-layer":
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])
        else:
            raise NotImplementedError


class AE(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        normalize: bool = True,
        simplex_dim: int = 10,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.mlp_dims = mlp_dims
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            normalize=normalize,
            simplex_dim=simplex_dim,
        )
        self.decoder = Decoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

    def reset(self, reset_type: str = "last-layer"):
        logger.info("Resetting encoder/decoder params")
        self.encoder.reset(reset_type=reset_type)
        self.decoder.reset(reset_type=reset_type)


class DDPG_AE(Agent):
    def __init__(
        self,
        # DDPG config
        observation_space: Space,
        action_space: Box,
        mlp_dims: List[int] = [512, 512],
        # exploration_noise: float = 0.2,
        exploration_noise_start: float = 1.0,
        exploration_noise_end: float = 0.1,
        exploration_noise_num_steps: int = 50,  # number of episodes do decay noise
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        utd_ratio: int = 1,  # DDPG parameter update-to-data ratio
        actor_update_freq: int = 1,  # update actor less frequently than critic
        # nstep: int = 1,
        discount: float = 0.99,
        tau: float = 0.005,
        act_with_target: bool = False,  # if True act with target network
        # Reset stuff
        reset_type: str = "last_layer",  # "full" or "last-layer"
        reset_strategy: str = "latent-dist",  #  "latent-dist" or "every-x-param-updates"
        reset_params_freq: int = 100000,  # reset params after this many param updates
        reset_threshold: float = 0.01,
        memory_size: int = 10000,
        # AE config
        train_strategy: str = "interleaved",  # "interleaved" or "representation-first"
        temporal_consistency: bool = False,  # if True include dynamic model for representation learning
        reward_loss: bool = False,  # if True include reward model for representation learning
        value_dynamics_loss: bool = False,  # if True include value prediction for representation learning
        value_enc_loss: bool = False,  # if True include value prediction for representation learning
        reconstruction_loss: bool = True,  # if True use reconstruction loss with decoder
        ae_learning_rate: float = 3e-4,
        ae_batch_size: int = 128,
        # ae_num_updates: int = 1000,
        ae_utd_ratio: int = 1,  # used for representation first training
        ae_patience: int = 100,
        ae_min_delta: float = 0.0,
        latent_dim: int = 20,
        ae_tau: float = 0.005,
        ae_normalize: bool = True,
        simplex_dim: int = 10,
        # encoder_reset_params_freq: int = 10000,  # reset enc params after X param updates
        device: str = "cuda",
        name: str = "DDPG-AE",
    ):
        # super().__init__(
        #     observation_space=observation_space, action_space=action_space, name=name
        # )
        self.train_strategy = train_strategy

        self.temporal_consistency = temporal_consistency
        self.reward_loss = reward_loss
        self.value_dynamics_loss = value_dynamics_loss
        self.value_enc_loss = value_enc_loss
        self.reconstruction_loss = reconstruction_loss
        self.ae_learning_rate = ae_learning_rate
        self.ae_batch_size = ae_batch_size
        # self.ae_num_updates = ae_num_updates
        self.ae_utd_ratio = ae_utd_ratio
        self.ae_patience = ae_patience
        self.ae_min_delta = ae_min_delta
        self.latent_dim = latent_dim
        self.ae_tau = ae_tau
        self.ae_normalize = ae_normalize
        self.simplex_dim = simplex_dim

        self.device = device

        if temporal_consistency or value_dynamics_loss:
            self.dynamics = MLPDynamics(
                action_space=action_space,
                mlp_dims=mlp_dims,
                latent_dim=latent_dim,
                normalize=ae_normalize,
                simplex_dim=simplex_dim,
            ).to(device)
        if self.reward_loss:
            self.reward = MLPReward(
                action_space=action_space,
                mlp_dims=mlp_dims,
                latent_dim=latent_dim,
            ).to(device)

        self.ae = AE(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            normalize=ae_normalize,
            simplex_dim=simplex_dim,
        ).to(device)
        self.ae_target = AE(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            normalize=ae_normalize,
            simplex_dim=simplex_dim,
        ).to(device)
        encoder_params = list(self.ae.parameters())
        if temporal_consistency:
            encoder_params += list(self.dynamics.parameters())
        self.ae_opt = torch.optim.AdamW(encoder_params, lr=ae_learning_rate)
        self.ae_target.load_state_dict(self.ae.state_dict())

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
        # Init DDPG agent
        self.ddpg = agents.DDPG(
            observation_space=self.latent_observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
            # act_fn=act_fn,
            # exploration_noise=exploration_noise,
            exploration_noise_start=exploration_noise_start,
            exploration_noise_end=exploration_noise_end,
            exploration_noise_num_steps=exploration_noise_num_steps,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            learning_rate=learning_rate,
            batch_size=batch_size,
            utd_ratio=utd_ratio,
            actor_update_freq=actor_update_freq,
            # reset_params_freq=reset_params_freq,
            reset_params_freq=None,  # handle resetting in this class
            reset_type=None,  # handle resetting in this class
            # nstep=nstep,  # N-step returns for critic training
            discount=discount,
            tau=tau,
            device=device,
            act_with_target=act_with_target,
            name=name,
        )

        # self.use_memory = False
        # self.use_memory_flag = False

        self.encoder_update_conter = 0

        self.reset_type = reset_type
        self.reset_strategy = reset_strategy
        # self.encoder_reset_params_freq = encoder_reset_params_freq
        self.reset_params_freq = reset_params_freq
        self.reset_threshold = reset_threshold
        self.memory_size = memory_size

        self.old_replay_buffer = None

        self.value_weight = 1
        self.value_weight_discount = 0.99999

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        self.ae.train()
        # self.ddpg.train()
        # self.ddpg.reset_flag = False

        # if self.reset_strategy == "latent-dist":
        #     if self.old_replay_buffer is None:
        #         logger.info("Building memory first time...")
        #         # logger.info("Updating memory...")
        #         self._update_memory(
        #             replay_buffer=replay_buffer, memory_size=self.memory_size
        #         )
        # update_memory = False
        # if self.reset_strategy == "latent-dist":
        #     if self.old_replay_buffer is None:
        #         logger.info("Building memory first time...")
        #         update_memory = True
        #         # self._update_memory(
        #         #     replay_buffer=replay_buffer, memory_size=self.memory_size
        #         # )
        #     elif self.trigger_reset():
        #         self.reset()
        #         update_memory = True
        #         # Do more updates if reset
        #         # TODO how many updates should be done when reset?
        #         num_new_transitions = replay_buffer.size()

        if self.train_strategy == "interleaved":
            info = self.update_1(
                replay_buffer=replay_buffer, num_new_transitions=num_new_transitions
            )
        elif self.train_strategy == "representation-first":
            info = self.update_2(
                replay_buffer=replay_buffer, num_new_transitions=num_new_transitions
            )
        else:
            raise NotImplementedError(
                "train_strategy should be either 'interleaved' or 'representation-first'"
            )

        self.ddpg._exploration_noise.step()
        self.ae.eval()
        # self.ddpg.eval()
        return info

    def update_1(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and DDPG at same time"""
        num_updates = int(num_new_transitions * self.ddpg.utd_ratio)
        info = {}

        if wandb.run is not None:
            wandb.log({"exploration_noise": self.ddpg.exploration_noise})

        logger.info(f"Performing {num_updates} DDPG-AE updates...")
        reset_flag = 0
        for i in range(num_updates):
            batch = replay_buffer.sample(self.ddpg.batch_size)
            # num_encoder_updates = 2  # 2 encoder updates for one q update
            # for i in range(num_encoder_updates):
            info.update(self.update_representation_step(batch=batch))

            # Map observations to latent
            latent_obs = self.ae_target.encoder(batch.observations)
            latent_next_obs = self.ae_target.encoder(batch.next_observations)
            # batch.observation = latent_obs.to(torch.float).detach()
            # batch.next_observation = latent_next_obs.to(torch.float).detach()
            latent_batch = ReplayBufferSamples(
                observations=latent_obs.to(torch.float).detach(),
                actions=batch.actions,
                next_observations=latent_next_obs.to(torch.float).detach(),
                dones=batch.dones,
                rewards=batch.rewards,
            )

            # DDPG on latent representation
            info.update(self.ddpg.update_step(batch=latent_batch))

            # Potentially reset ae/actor/critic NN params
            if self.reset_strategy == "every-x-param-updates":
                if self.reset_params_freq is not None:
                    if self.ddpg.critic_update_counter % self.reset_params_freq == 0:
                        logger.info(
                            f"Resetting as step {self.ddpg.critic_update_counter} % {self.reset_params_freq} == 0"
                        )
                        self.reset()
                        reset_flag = 1
            elif self.reset_strategy == "latent-dist":
                if self.trigger_reset_latent_dist(replay_buffer=replay_buffer):
                    reset_flag = 1
            else:
                raise NotImplementedError(
                    "reset_strategy should be either 'every-x-params-updates' or 'latent-dist'"
                )

            if reset_flag == 1:
                if wandb.run is not None:
                    wandb.log({"reset": reset_flag})

            if i % 100 == 0:
                logger.info(
                    f"Iteration {i} | loss {info['loss']} | rec loss {info['rec_loss']} | tc loss {info['temporal_consitency_loss']} | reward loss {info['reward_loss']} | value dynamics loss {info['value_dynamics_loss']}"
                )
                if wandb.run is not None:
                    # info.update({"exploration_noise": self.ddpg.exploration_noise})
                    wandb.log(info)
                    wandb.log({"reset": reset_flag})
                reset_flag = 0

        logger.info("Finished training DDPG-AE")

        return info

    def update_2(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and then do DDPG"""
        if self.ae_early_stopper is not None:
            self.ae_early_stopper.reset()

        num_ae_updates = int(num_new_transitions * self.ae_utd_ratio)

        reset_flag = 0
        if self.reset_strategy == "latent-dist":
            if self.trigger_reset_latent_dist(replay_buffer=replay_buffer):
                reset_flag = 1

        if wandb.run is not None:
            wandb.log({"exploration_noise": self.ddpg.exploration_noise})
            if reset_flag == 1:
                wandb.log({"reset": reset_flag})

        info = {}
        logger.info("Training AE...")
        for i in range(num_ae_updates):
            batch = replay_buffer.sample(self.ae_batch_size)
            info.update(self.update_representation_step(batch=batch))

            if i % 100 == 0:
                logger.info(
                    f"Iteration {i} | loss {info['loss']} | rec loss {info['rec_loss']} | tc loss {info['temporal_consitency_loss']} | reward loss {info['reward_loss']} | value dynamics loss {info['value_dynamics_loss']}"
                )
                if wandb.run is not None:
                    # info.update({"exploration_noise": self.ddpg.exploration_noise})
                    wandb.log(info)
                    wandb.log({"reset": reset_flag})
                # reset_flag = 0

            if self.ae_early_stopper is not None:
                # TODO this should use a validation loss
                if self.ae_early_stopper(info["loss"]):
                    logger.info("Early stopping criteria met, stopping AE training...")
                    break

        if self.reset_strategy == "latent-dist" and reset_flag == 1:
            logger.info("Updating memory...")
            self._update_memory(
                replay_buffer=replay_buffer, memory_size=self.memory_size
            )

        logger.info("Finished training AE")

        num_updates = num_new_transitions * self.ddpg.utd_ratio
        logger.info(f"Performing {num_updates} DDPG updates...")
        reset_flag = 0
        for i in range(num_updates):
            batch = replay_buffer.sample(self.ddpg.batch_size)

            # Map observations to latent
            latent_obs = self.ae_target.encoder(batch.observations)
            latent_next_obs = self.ae_target.encoder(batch.next_observations)
            # batch.observation = latent_obs.to(torch.float).detach()
            # batch.next_observation = latent_next_obs.to(torch.float).detach()
            latent_batch = ReplayBufferSamples(
                observations=latent_obs.to(torch.float).detach(),
                actions=batch.actions,
                next_observations=latent_next_obs.to(torch.float).detach(),
                dones=batch.dones,
                rewards=batch.rewards,
            )

            # DDPG on latent representation
            info.update(self.ddpg.update_step(batch=latent_batch))

            # Potentially reset ae/actor/critic NN params
            if self.reset_strategy == "every-x-param-updates":
                if self.reset_params_freq is not None:
                    if self.ddpg.critic_update_counter % self.reset_params_freq == 0:
                        logger.info(
                            f"Resetting as step {self.ddpg.critic_update_counter} % {self.reset_params_freq} == 0"
                        )
                        self.reset()
                        reset_flag = 1
                        if wandb.run is not None:
                            wandb.log({"reset": reset_flag})

            if i % 100 == 0:
                if wandb.run is not None:
                    wandb.log(info)
                reset_flag = 0

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

        x_train = batch.observations
        if self.reconstruction_loss:
            x_rec, z = self.ae(x_train)
            rec_loss = (x_rec - x_train).abs().mean()
        else:
            z = self.ae.encoder(x_train)
            rec_loss = torch.zeros(1).to(self.device)

        if self.temporal_consistency:
            # TODO multistep temporal consistency
            delta_z_dynamics = self.dynamics(x=z, a=batch.actions)
            z_next_dynamics = z + delta_z_dynamics
            with torch.no_grad():
                z_next_enc_target = self.ae_target.encoder(batch.next_observations)
            temporal_consitency_loss = torch.nn.functional.mse_loss(
                input=z_next_enc_target, target=z_next_dynamics, reduction="mean"
            )
        else:
            temporal_consitency_loss = torch.zeros(1).to(self.device)

        if self.reward_loss:
            reward_pred = self.reward(z=z, a=batch.actions)
            reward_loss = torch.nn.functional.mse_loss(
                input=batch.rewards, target=reward_pred, reduction="mean"
            )
        else:
            reward_loss = torch.zeros(1).to(self.device)

        def value_loss_fn(z_next):
            q1_pred, q2_pred = self.ddpg.critic(z, batch.actions)

            # Policy smoothing actions for next state
            # TODO get this functionality from method in ddpg
            clipped_noise = (
                torch.randn_like(batch.actions, device=self.device)
                * self.ddpg.policy_noise
            ).clamp(
                -self.ddpg.noise_clip, self.ddpg.noise_clip
            ) * self.ddpg.target_actor.action_scale

            next_state_actions = (self.ddpg.target_actor(z_next) + clipped_noise).clamp(
                self.ddpg.action_space.low[0], self.ddpg.action_space.high[0]
            )
            q1_next_target, q2_next_target = self.ddpg.target_critic(
                z_next, next_state_actions
            )
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            next_q_value = batch.rewards.flatten() + (
                1 - batch.dones.flatten()
            ) * self.ddpg.discount**self.ddpg.nstep * (min_q_next_target).view(-1)
            q1_loss = torch.nn.functional.mse_loss(
                input=q1_pred, target=next_q_value, reduction="mean"
            )
            q2_loss = torch.nn.functional.mse_loss(
                input=q2_pred, target=next_q_value, reduction="mean"
            )
            value_loss = (q1_loss + q2_loss) / 2
            return value_loss

        if self.value_dynamics_loss:
            if not self.temporal_consistency:
                delta_z_dynamics = self.dynamics(x=z, a=batch.actions)
                z_next_dynamics = z + delta_z_dynamics
            value_dynamics_loss = value_loss_fn(z_next_dynamics)
        else:
            value_dynamics_loss = torch.zeros(1).to(self.device)

        if self.value_enc_loss:
            if not self.temporal_consistency:
                with torch.no_grad():
                    z_next_enc_target = self.ae_target.encoder(batch.next_observations)
            value_enc_loss = value_loss_fn(z_next_enc_target)
        else:
            value_enc_loss = torch.zeros(1).to(self.device)

        # if (self.ddpg.critic_update_counter / 1000) > 5:
        #     print(
        #         f"(self.ddpg.critic_update_counter / 1000) {(self.ddpg.critic_update_counter / 1000)}"
        #     )
        #     self.value_weight *= self.value_weight_discount

        loss = (
            rec_loss
            + reward_loss
            + temporal_consitency_loss
            # + self.value_weight * temporal_consitency_loss
            # + (1 - self.value_weight) * value_dynamics_loss
            # + (1 - self.value_weight) * value_enc_loss
            + value_dynamics_loss
            + value_enc_loss
        )
        info = {
            "reward_loss": reward_loss.item(),
            "value_dynamics_loss": value_dynamics_loss.item(),
            "value_enc_loss": value_enc_loss.item(),
            # "value_weight": self.value_weight,
            "temporal_consitency_loss": temporal_consitency_loss.item(),
            "rec_loss": rec_loss.item(),
            "loss": loss.item(),
        }

        self.ae_opt.zero_grad()
        loss.backward()
        self.ae_opt.step()

        # Update the target network
        soft_update_params(self.ae, self.ae_target, tau=self.ae_tau)

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

    def _update_memory(self, replay_buffer: ReplayBuffer, memory_size: int = 10000):
        # self.use_memory = True
        # if self.use_memory:
        #     self.use_memory_flag = True
        # memory_size = replay_buffer.size()

        self.old_replay_buffer = replay_buffer

        self.old_ae = AE(
            observation_space=self.ae.observation_space,
            mlp_dims=self.ae.mlp_dims,
            latent_dim=self.ae.latent_dim,
            normalize=self.ae_normalize,
            simplex_dim=self.simplex_dim,
        ).to(self.device)
        self.old_ae.load_state_dict(self.ae.state_dict())

        # memory = replay_buffer.sample(memory_size)
        # print(f"memory {memory}")
        # self.x_mem = torch.concat([memory.observations, memory.next_observations], 0)
        memory_batch = self.old_replay_buffer.sample(memory_size)
        self.x_mem = memory_batch.observations
        print(f"self.x_mem {self.x_mem.shape}")
        with torch.no_grad():
            self.z_mem = self.old_ae.encoder(self.x_mem)
        print(f"self.z_mem {self.z_mem.shape}")
        self.z_mem_max = torch.max(self.z_mem, 0)[0]
        self.z_mem_min = torch.min(self.z_mem, 0)[0]
        self.z_mem_normalised = (self.z_mem - self.z_mem_min) / (
            self.z_mem_max - self.z_mem_min
        )

    # def trigger_reset(self) -> dict:
    #     """Returns 1 if it has reset and 0 otherwise"""
    #     reset, mem_dist = 0, 0
    #     if self.reset_strategy == "every-x-param-updates":
    #         if self.reset_params_freq is not None:
    #             if self.ddpg.critic_update_counter % self.reset_params_freq == 0:
    #                 logger.info(
    #                     f"Resetting as step {self.ddpg.critic_update_counter} % {self.reset_params_freq} == 0"
    #                 )
    #                 self.reset(full_reset=False)
    #                 reset = 1
    #     elif self.reset_strategy == "latent-dist":
    #         z_mem_pred = self.ae.encoder(self.x_mem)
    #         # TODO make sure mean is over state dimensions
    #         mem_dist = (self.z_mem - z_mem_pred).abs().mean()
    #         logger.info(f"mem_dist {mem_dist}")
    #         if mem_dist > self.reset_threshold:
    #             logger.info(
    #                 f"Resetting as mem_dist {mem_dist} > reset_threshold {self.reset_threshold}"
    #             )
    #             self.reset(full_reset=False)
    #             reset = 1
    #     else:
    #         raise NotImplementedError(
    #             "reset_strategy should be either 'every-x-params-updates' or 'latent-dist'"
    #         )
    #     return {"reset": reset, "mem_dist": mem_dist}

    def trigger_reset_latent_dist(self, replay_buffer) -> bool:
        if self.old_replay_buffer is None:
            logger.info("Building memory first time...")
            # logger.info("Updating memory...")
            self._update_memory(
                replay_buffer=replay_buffer, memory_size=self.memory_size
            )
            return False
        z_mem_pred = self.ae.encoder(self.x_mem)
        z_mem_pred_normalised = (z_mem_pred - self.z_mem_min) / (
            self.z_mem_max - self.z_mem_min
        )
        # TODO make sure mean is over state dimensions
        mem_dist = (self.z_mem_normalised - z_mem_pred_normalised).abs().mean()
        # mem_dist = (self.z_mem - z_mem_pred).abs().mean()
        logger.info(f"mem_dist {mem_dist}")
        # breakpoint()
        if wandb.run is not None:
            wandb.log({"mem_dist": mem_dist})
        if mem_dist > self.reset_threshold:
            reset = True
            logger.info(
                f"Resetting as mem_dist {mem_dist} > reset_threshold {self.reset_threshold}"
            )
            self.reset()
        else:
            reset = False
        return reset

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        observation = torch.Tensor(observation).to(self.device)
        _, z = self.ae(observation)
        z = z.to(torch.float)
        return self.ddpg.select_action(observation=z, eval_mode=eval_mode, t0=t0)

    def reset(self, reset_type: Optional[str] = None):
        if reset_type is None:
            reset_type = self.reset_type
        logger.info("Restting agent...")
        self.ae.reset(reset_type=reset_type)
        if self.temporal_consistency or self.value_dynamics_loss:
            self.dynamics.reset(reset_type=reset_type)
        if self.reward_loss:
            self.reward.reset(reset_type=reset_type)
        # self.ae_target.reset(full_reset=full_reset)
        self.ae_target.load_state_dict(self.ae.state_dict())
        self.ae_opt = torch.optim.AdamW(
            list(self.ae.parameters()), lr=self.ae_learning_rate
        )

        self.ddpg.reset(reset_type=reset_type)

        self.ae_early_stopper = h.EarlyStopper(
            patience=self.ae_patience, min_delta=self.ae_min_delta
        )
