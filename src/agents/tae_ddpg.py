#!/usr/bin/env python3
import logging
from typing import Any, List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gymnasium as gym
import numpy as np
import src
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from gymnasium.spaces import Box, Space
from src.agents.utils import EarlyStopper
from src.custom_types import Action, EvalMode, State, T0
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from vector_quantize_pytorch import FSQ, VectorQuantize

from .agent import Agent


class Encoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        act_fn=nn.ELU,
    ):
        super().__init__()
        in_dim = np.array(observation_space.shape).prod()
        # TODO data should be normalized???
        self.latent_dim = latent_dim
        self._mlp = src.agents.utils.mlp(
            in_dim=in_dim, mlp_dims=mlp_dims, out_dim=latent_dim, act_fn=act_fn
        )
        self.apply(src.agents.utils.orthogonal_init)

    def forward(self, x):
        z = self._mlp(x)
        return z


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
        self._mlp = src.agents.utils.mlp(
            in_dim=latent_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=act_fn
        )
        self.apply(src.agents.utils.orthogonal_init)

    def forward(self, z):
        x = self._mlp(z)
        return x


class MLPDynamics(nn.Module):
    def __init__(
        self,
        action_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        act_fn=nn.ELU,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = np.array(action_space.shape).prod() + latent_dim
        # TODO data should be normalized???
        self._mlp = src.agents.utils.mlp(
            in_dim=in_dim, mlp_dims=mlp_dims, out_dim=latent_dim + 1, act_fn=act_fn
        )
        self.apply(src.agents.utils.orthogonal_init)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        z = self._mlp(x)
        r = z[..., -1:]
        return z[..., :-1], r


class VAE(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        act_fn=nn.ELU,
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


class TAEDDPG(Agent):
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
        num_updates: int = 1000,
        # VAE config
        vae_learning_rate: float = 3e-4,
        vae_batch_size: int = 128,
        vae_num_updates: int = 1000,
        vae_patience: int = 100,
        vae_min_delta: float = 0.0,
        latent_dim: int = 20,
        vae_tau: float = 0.005,
        # actor_update_freq: int = 2,  # update actor less frequently than critic
        # nstep: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
        name: str = "DDPG",
    ):
        # super().__init__(
        #     observation_space=observation_space, action_space=action_space, name=name
        # )
        self.vae_batch_size = vae_batch_size
        self.vae_learning_rate = vae_learning_rate
        self.vae_num_updates = vae_num_updates
        self.latent_dim = latent_dim

        self.vae_tau = vae_tau

        self.device = device

        self.dynamics = MLPDynamics(
            action_space=action_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        )
        self.vae = VAE(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        ).to(device)
        self.vae_target = VAE(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        ).to(device)
        self.vae_opt = torch.optim.AdamW(
            list(self.vae.parameters()) + list(self.dynamics.parameters()),
            lr=vae_learning_rate,
        )
        self.vae_target.load_state_dict(self.vae.state_dict())

        self.vae_patience = vae_patience
        self.vae_min_delta = vae_min_delta

        # TODO make a space for latent states
        # latent_observation_space = observation_space
        # high = np.array(levels).prod()
        # TODO is this the right way to make observation space??
        # TODO Should we bound z in -100,100 instead of -inf,inf??
        self.latent_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(latent_dim,)
            # low=0.0, high=high, shape=(latent_dim,)
        )
        print(f"latent_observation_space {self.latent_observation_space}")
        # Init ddpg agent
        self.ddpg = src.agents.DDPG(
            observation_space=self.latent_observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
            act_fn=act_fn,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_updates=num_updates,
            gamma=gamma,
            tau=tau,
            device=device,
            name=name,
        )

    def train(
        self,
        replay_buffer: ReplayBuffer,
        num_updates: Optional[int] = None,
        reinit_opts: bool = False,
    ) -> dict:
        if reinit_opts:
            raise NotImplementedError
        logger.info("Training representation (encoder/dynamics)...")
        info = self.train_encoder_and_dynamics(replay_buffer=replay_buffer)
        # info = self.train_vae_vae(replay_buffer=replay_buffer, num_updates=num_updates)
        logger.info("Finished training representation (encoder/dynamics)")

        vae_target = self.vae_target

        class LatentReplayBuffer:
            def sample(self, batch_size: int):
                batch = replay_buffer.sample(batch_size=batch_size)
                latent_obs = vae_target.encoder(batch.observations)
                latent_next_obs = vae_target.encoder(batch.next_observations)
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
        logger.info("Training DDPG...")
        info.update(
            self.ddpg.train(replay_buffer=latent_replay_buffer, num_updates=num_updates)
        )
        logger.info("Finished training DDPG")
        return info

    def train_encoder_and_dynamics(
        self, replay_buffer: ReplayBuffer, num_updates: Optional[int] = None
    ):
        if num_updates is None:
            num_updates = self.vae_num_updates
        vae_early_stopper = EarlyStopper(
            patience=self.vae_patience, min_delta=self.vae_min_delta
        )
        info = {}
        i = 0
        for _ in range(num_updates):
            batch = replay_buffer.sample(self.vae_batch_size)
            x_rec, z = self.vae(batch.observations)
            x_rec_next, z_next_enc = self.vae(batch.next_observations)
            x_rec_next_target, z_next_enc_target = self.vae_target(
                batch.next_observations
            )
            z_next_dynamics, reward_dynamics = self.dynamics(x=z, a=batch.actions)
            # breakpoint()
            latent_state_consitency_loss = torch.nn.functional.mse_loss(
                input=z_next_enc_target,
                target=z_next_dynamics,
                reduction="mean"
                # input=z_next_enc, target=z_next_dynamics, reduction="mean"
            )
            reward_loss = torch.nn.functional.mse_loss(
                input=batch.rewards, target=reward_dynamics, reduction="mean"
            )
            rec_loss = (x_rec - batch.observations).abs().mean()
            rec_next_loss = (x_rec_next - batch.next_observations).abs().mean()
            loss = latent_state_consitency_loss
            # loss = latent_state_consitency_loss + reward_loss
            self.vae_opt.zero_grad()
            loss.backward()
            # rec_loss.backward()
            self.vae_opt.step()

            # Update the target network
            for param, target_param in zip(
                self.vae.parameters(), self.vae_target.parameters()
            ):
                target_param.data.copy_(
                    self.vae_tau * param.data + (1 - self.vae_tau) * target_param.data
                )

            if i % 100 == 0:
                print(f"Iteration {i} loss {loss}")
                try:
                    wandb.log(
                        {
                            "rec_loss": rec_loss.item(),
                            "rec_next_loss": rec_next_loss.item(),
                            "latent_state_consitency_loss": latent_state_consitency_loss.item(),
                            "reward_loss": reward_loss.item(),
                            "loss": loss.item(),
                        }
                    )
                except:
                    pass
            i += 1
            if vae_early_stopper(loss):
                logger.info("Early stopping criteria met, stopping VAE training...")
                break
        return info

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        observation = torch.Tensor(observation).to(self.device)
        _, z = self.vae(observation)
        z = z.to(torch.float)
        # indices = torch.flatten(indices, -len(self.levels), -1)
        # z= torch.flatten(z, -2, -1)
        return self.ddpg.select_action(observation=z, eval_mode=eval_mode, t0=t0)
