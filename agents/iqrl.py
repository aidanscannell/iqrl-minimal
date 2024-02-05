#!/usr/bin/env python3
import logging
from typing import List, Optional, Tuple

import agents
import gymnasium as gym
import helper as h
import numpy as np
import torch
import torch.nn as nn
import wandb
from custom_types import Agent, EvalMode, T0
from gymnasium.spaces import Box, Space
from utils import ReplayBuffer, ReplayBufferSamples


logger = logging.getLogger(__name__)


class iQRL(Agent):
    def __init__(
        self,
        ##### TD3 config #####
        observation_space: Space,
        action_space: Box,
        mlp_dims: List[int] = [512, 512],
        exploration_noise_start: float = 1.0,
        exploration_noise_end: float = 0.1,
        exploration_noise_num_steps: int = 50,  # number of episodes do decay noise
        policy_noise: float = 0.2,  # for policy smoothing
        noise_clip: float = 0.3,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        utd_ratio: int = 1,  # TD3 parameter update-to-data ratio
        actor_update_freq: int = 2,  # update actor less frequently than critic
        nstep: int = 1,  # nstep used for TD returns
        gamma: float = 0.99,
        tau: float = 0.005,
        act_with_tar: bool = False,  # if True act with tar actor network
        ##### Encoder config #####
        latent_dim: int = 128,
        horizon: int = 5,  # horizon used for representation learning
        rho: float = 0.9,  # gamma for multi-step representation loss
        enc_mlp_dims: List[int] = [256],
        enc_learning_rate: float = 3e-4,
        enc_tau: float = 0.005,  # momentum coefficient for target encoder
        enc_batch_size: int = 128,
        enc_utd_ratio: int = 1,  # used for representation first training
        enc_update_freq: int = 1,  # update enc less frequently than actor/critic
        ##### Configure which loss terms to use #####
        use_tc_loss: bool = True,  # if True include dynamic model for representation learning
        use_rew_loss: bool = False,  # if True include reward model for representation learning
        use_rec_loss: bool = False,  # if True use reconstruction loss with decoder
        use_cosine_similarity_dynamics: bool = False,
        use_cosine_similarity_reward: bool = False,
        # Project loss into another space before calculating TC loss
        use_project_latent: bool = False,
        projection_mlp_dims: List[int] = [256],
        projection_dim: Optional[int] = None,  # if None use latent_dim/2
        ##### Configure where we use the target encoder network #####
        use_tar_enc: bool = True,  # if True use target enc for actor/critic
        act_with_tar_enc: bool = False,  # if True act with target enc network
        ##### Configure FSQ normalization #####
        use_fsq: bool = False,
        fsq_levels: List[int] = [8, 6, 5],
        quantized: bool = False,  # if True use quantized latent for TD3, else use normalized latent
        # fsq_idx: int = 0,  # 0 uses z and 1 uses indices for actor/critic
        ##### Other stuff #####
        logging_freq: int = 499,
        compile: bool = False,
        device: str = "cuda",
        name: str = "iQRL",
    ):
        super().__init__(
            observation_space=observation_space, action_space=action_space, name=name
        )

        ##### Encoder config #####
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.rho = rho
        self.enc_learning_rate = enc_learning_rate
        self.enc_tau = enc_tau
        self.enc_batch_size = enc_batch_size
        self.enc_utd_ratio = enc_utd_ratio
        self.enc_update_freq = enc_update_freq

        ##### Configure which loss terms to use #####
        self.use_tc_loss = use_tc_loss
        self.use_rew_loss = use_rew_loss
        self.use_rec_loss = use_rec_loss
        self.use_cosine_similarity_dynamics = use_cosine_similarity_dynamics
        self.use_cosine_similarity_reward = use_cosine_similarity_reward

        # Project loss into another space before calculating TC loss
        self.use_project_latent = use_project_latent

        ##### Configure where we use the target encoder network #####
        self.use_tar_enc = use_tar_enc
        self.act_with_tar_enc = act_with_tar_enc

        ##### Encoder config #####
        self.use_fsq = use_fsq
        self.quantized = quantized
        if quantized:
            self.fsq_idx = 1
        else:
            self.fsq_idx = 0

        ##### Other stuff #####
        self.logging_freq = logging_freq
        self.device = device

        ##### Calculate dimensions for MLPs #####
        obs_dim = np.array(observation_space.shape).prod()
        state_act_dim = self.latent_dim + np.array(action_space.shape).prod()

        ##### Init encoder for representation learning #####
        if use_fsq:
            self.enc = h.FSQMLP(
                in_dim=obs_dim,
                mlp_dims=enc_mlp_dims,
                levels=fsq_levels,
                out_dim=latent_dim,
            ).to(device)
            self.enc_tar = h.FSQMLP(
                in_dim=obs_dim,
                mlp_dims=enc_mlp_dims,
                levels=fsq_levels,
                out_dim=latent_dim,
            ).to(device)
        else:
            self.enc = h.mlp(in_dim=obs_dim, mlp_dims=mlp_dims, out_dim=latent_dim).to(
                device
            )
            self.enc_tar = h.mlp(
                in_dim=obs_dim, mlp_dims=mlp_dims, out_dim=latent_dim
            ).to(device)
        self.enc_tar.load_state_dict(self.enc.state_dict())
        if compile:
            self.enc = torch.compile(self.enc, mode="default")
            self.enc_tar = torch.compile(self.enc_tar, mode="default")
        enc_params = list(self.enc.parameters())

        ##### (Optionally) Init decoder for representation learning  ####
        if use_rec_loss:
            self.decoder = h.mlp(
                in_dim=latent_dim, mlp_dims=mlp_dims, out_dim=obs_dim
            ).to(device)
            if compile:
                self.decoder = torch.compile(self.decoder, mode="default")
            enc_params += list(self.decoder.parameters())

        ##### (Optionally) Init dynamics for representation learning  ####
        if use_tc_loss:
            if use_fsq:
                self.dynamics = h.FSQMLP(
                    in_dim=state_act_dim,
                    mlp_dims=mlp_dims,
                    levels=fsq_levels,
                    out_dim=latent_dim,
                ).to(device)
            else:
                self.dynamics = h.mlp(
                    in_dim=state_act_dim, mlp_dims=mlp_dims, out_dim=latent_dim
                ).to(device)
            if compile:
                self.dynamics = torch.compile(self.dynamics, mode="default")
            enc_params += list(self.dynamics.parameters())

        ##### (Optionally) Init reward model for representation learning  ####
        if self.use_rew_loss:
            self.reward = h.mlp(in_dim=state_act_dim, mlp_dims=mlp_dims, out_dim=1)
            self.reward.to(device)
            if compile:
                self.reward = torch.compile(self.reward, mode="default")
            enc_params += list(self.reward.parameters())

        # (Optionally) Init projection MLP to calculate TC loss in projected space
        if self.use_project_latent:
            if projection_dim is None:
                projection_dim = int(latent_dim / 2)
            self.projection = h.mlp(
                in_dim=latent_dim, mlp_dims=projection_mlp_dims, out_dim=projection_dim
            ).to(device)
            self.projection_tar = h.mlp(
                in_dim=latent_dim, mlp_dims=projection_mlp_dims, out_dim=projection_dim
            ).to(device)
            self.projection_tar.load_state_dict(self.projection.state_dict())
            if compile:
                self.projection = torch.compile(self.projection, mode="default")
                self.projection_tar = torch.compile(self.projection_tar, mode="default")
            enc_params += list(self.projection.parameters())

        ##### Init optimizer for representation learning #####
        self.enc_opt = torch.optim.AdamW(enc_params, lr=enc_learning_rate)

        ##### Make latent observation space #####
        num_levels = len(fsq_levels)
        obs_low = -np.inf
        obs_high = np.inf
        obs_shape = (latent_dim,)
        if use_fsq:
            if self.quantized:
                obs_low = 1
                obs_high = np.array(fsq_levels).prod()
                obs_shape = (int(latent_dim / len(fsq_levels)),)
            else:
                obs_low = -num_levels / 2
                obs_high = num_levels / 2
        self.latent_observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=obs_shape
        )

        ##### Init TD3 agent #####
        self.td3 = agents.TD3(
            observation_space=self.latent_observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
            exploration_noise_start=exploration_noise_start,
            exploration_noise_end=exploration_noise_end,
            exploration_noise_num_steps=exploration_noise_num_steps,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            learning_rate=learning_rate,
            batch_size=batch_size,
            utd_ratio=utd_ratio,
            actor_update_freq=actor_update_freq,
            reset_params_freq=None,  # handle resetting in this class
            reset_type=None,  # handle resetting in this class
            nstep=nstep,  # N-step returns for critic training
            gamma=gamma,
            tau=tau,
            device=device,
            act_with_tar=act_with_tar,
            name=name,
            compile=compile,
        )

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and TD3 at same time"""
        self.train()

        num_updates = int(num_new_transitions * self.td3.utd_ratio)
        info = {}

        logger.info(f"Performing {num_updates} iQRL updates...")
        for i in range(num_updates):
            batch = replay_buffer.sample(self.td3.batch_size, val=False)
            # Update enc less frequently than actor/critic
            if i % self.enc_update_freq == 0:
                info.update(self.update_representation_step(batch=batch))

            # Map observations to latent
            with torch.no_grad():
                if self.use_tar_enc:
                    z = self.enc_tar(batch.observations)
                    z_next = self.enc_tar(batch.next_observations)
                else:
                    z = self.enc(batch.observations)
                    z_next = self.enc(batch.next_observations)

            # TD3 on latent representation
            info.update(
                self.td3.update_step(
                    batch=ReplayBufferSamples(
                        observations=z.to(torch.float).detach(),
                        actions=batch.actions,
                        next_observations=z_next.to(torch.float).detach(),
                        dones=batch.dones,
                        timeouts=batch.timeouts,
                        rewards=batch.rewards,
                        next_state_gammas=batch.next_state_gammas,
                    )
                )
            )

            if i % self.logging_freq == 0:
                logger.info(
                    f"Iteration {i} | loss {info['enc_loss']} | rec loss {info['rec_loss']} | tc loss {info['tc_loss']} | reward loss {info['reward_loss']} "
                )
                if wandb.run is not None:
                    wandb.log(info)

        logger.info("Finished training iQRL")

        ###### Log some stuff ######
        if wandb.run is not None:
            wandb.log({"exploration_noise": self.td3.exploration_noise})
            wandb.log({"buffer_size": replay_buffer.size()})

        self.td3.exploration_noise_schedule.step()
        self.eval()
        return info

    def update_representation_step(self, batch: ReplayBufferSamples):
        ##### Calculate loss #####
        loss, info = self.representation_loss(batch=batch)

        ##### Update parameters #####
        self.enc_opt.zero_grad()
        loss.backward()
        self.enc_opt.step()

        ##### Update the target network #####
        h.soft_update_params(self.enc, self.enc_tar, tau=self.enc_tau)
        if self.use_project_latent:
            h.soft_update_params(self.projection, self.projection_tar, tau=self.enc_tau)

        return info

    def representation_loss(
        self, batch: ReplayBufferSamples
    ) -> Tuple[torch.Tensor, dict]:
        tc_loss = torch.zeros(1).to(self.device)
        reward_loss = torch.zeros(1).to(self.device)
        rec_loss = torch.zeros(1).to(self.device)

        # Calculate the tc_loss (and reward) using multi-step dynamics predictions
        if self.use_tc_loss:
            z = self.enc(batch.observations[0])

            dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
            timeout_or_dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
            for t in range(self.horizon):
                # Truncate multi-step loss if episode is done
                dones = torch.where(timeout_or_dones, dones, batch.dones[t])
                timeout_or_dones = torch.logical_or(
                    timeout_or_dones, torch.logical_or(dones, batch.timeouts[t])
                )

                # Predict next latent (and reward)
                next_z_pred = z + self.dynamics(torch.concat([z, batch.actions[t]], -1))
                if self.use_rew_loss:
                    r_pred = self.reward(torch.concat([z, batch.actions[t]], -1))

                # Get target next_latent (and reward)
                with torch.no_grad():
                    next_obs = batch.next_observations[t]
                    next_z_tar = self.enc_tar(next_obs)
                    r_tar = batch.rewards[t]
                    assert next_obs.ndim == r_tar.ndim == 2

                # Calculate reconstruction loss for each time step
                if self.use_rec_loss:
                    x_rec = self.decoder(z)
                    # TODO this assumes x_train is of shape [H, B, D]
                    _rec_loss = torch.mean((x_rec - batch.observations[t]) ** 2, dim=-1)
                    rec_loss += self.rho**t * torch.mean(
                        (1 - timeout_or_dones.to(torch.int)) * _rec_loss
                    )

                # Don't forget this
                z = next_z_pred

                if self.use_project_latent:
                    next_z_tar = self.projection_tar(next_z_tar)
                    next_z_pred = self.projection(next_z_pred)

                # Losses
                if self.use_cosine_similarity_dynamics:
                    _tc_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)(
                        next_z_pred, next_z_tar
                    )
                else:
                    _tc_loss = torch.mean((next_z_pred - next_z_tar) ** 2, dim=-1)
                tc_loss += self.rho**t * torch.mean(
                    (1 - timeout_or_dones.to(torch.int)) * _tc_loss
                )
                if self.use_rew_loss:
                    assert r_pred.ndim == 2
                    assert r_tar.ndim == 2
                    if self.use_cosine_similarity_reward:
                        _reward_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)(
                            r_pred, r_tar
                        )
                    else:
                        _reward_loss = (r_pred[..., 0] - r_tar[..., 0]) ** 2
                    reward_loss += self.rho**t * torch.mean(
                        (1 - timeout_or_dones.to(torch.int)) * _reward_loss
                    )

        # Calculate reconstruction loss at each time step in horizon
        if self.use_rec_loss and not self.use_tc_loss:

            def rec_loss_fn(x):
                z = self.enc(x)
                x_rec = self.decoder(z)
                rec_loss = ((x_rec - x) ** 2).mean()
                return rec_loss

            rec_loss = torch.mean(torch.func.vmap(rec_loss_fn)(batch.observations))

        loss = rec_loss + reward_loss + tc_loss

        z = self.enc(batch.observations[0])
        info = {
            "reward_loss": reward_loss.item(),
            "tc_loss": tc_loss.item(),
            "rec_loss": rec_loss.item(),
            "enc_loss": loss.item(),
            "z_min": torch.min(z).item(),
            "z_max": torch.max(z).item(),
            "z_mean": torch.mean(z.to(torch.float)).item(),
            "z_median": torch.median(z).item(),
        }
        return loss, info

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        flag = False
        if observation.ndim > 2:
            if observation.shape[0] == 1:
                observation = observation[0, ...]
                flag = True
            else:
                raise NotImplementedError
        observation = torch.Tensor(observation).to(self.device)
        if self.fsq_idx == 1:
            raise NotImplementedError
        if self.act_with_tar_enc:
            z = self.enc_tar(observation)
        else:
            z = self.enc(observation)
        z = z.to(torch.float)

        action = self.td3.select_action(observation=z, eval_mode=eval_mode, t0=t0)
        if flag:
            action = action[None, ...]
        return action

    def train(self):
        self.enc.train()
        self.td3.train()
        if self.use_rec_loss:
            self.decoder.train()
        if self.use_tc_loss:
            self.dynamics.train()
        if self.use_rew_loss:
            self.reward.train()

    def eval(self):
        self.enc.eval()
        self.td3.eval()
        if self.use_rec_loss:
            self.decoder.eval()
        if self.use_tc_loss:
            self.dynamics.eval()
        if self.use_rew_loss:
            self.reward.eval()
