#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from typing import List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import src
import torch
import torch.nn as nn
import wandb
from gymnasium.spaces import Space
from src.agents.utils import EarlyStopper, soft_update_params
from src.custom_types import BatchLatent, BatchObservation
from stable_baselines3.common.buffers import ReplayBuffer


# @dataclass
@dataclass(eq=False)
# class Encoder(nn.Module):
class Encoder:
    latent_dim: int

    def __call__(
        self, observation: BatchObservation, target: bool = False
    ) -> BatchLatent:
        raise NotImplementedError

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        raise NotImplementedError

    # def __post_init__(self):
    #     super().__init__()


class AE(Encoder):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        act_fn=nn.ELU,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        utd_ratio: int = 1,
        tau: float = 0.005,
        encoder_reset_params_freq: int = 10000,  # reset enc params after X param updates
        early_stopper: Optional[EarlyStopper] = None,
        # early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)
        device: str = "cuda",
        name: str = "AE",
    ):
        super().__init__(latent_dim=latent_dim)
        self.observation_space = observation_space
        self.mlp_dims = mlp_dims
        self.act_fn = act_fn

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.tau = tau
        self.encoder_reset_params_freq = encoder_reset_params_freq
        self.early_stopper = early_stopper
        self.device = device
        self.name = name

        self.encoder = MLPEncoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        )
        self.target_encoder = MLPEncoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        )
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.decoder = MLPDecoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        )
        self.target_decoder = MLPDecoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        )
        self.target_decoder.load_state_dict(self.decoder.state_dict())

        self.opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )

    def __call__(
        self, observation: BatchObservation, target: bool = False
    ) -> BatchLatent:
        if target:
            z = self.target_encoder(observation)
        else:
            z = self.encoder(observation)
        return z

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int):
        if self.early_stopper is not None:
            self.early_stopper.reset()

        self.encoder.train()
        self.decoder.train()
        num_updates = num_new_transitions * self.utd_ratio

        for i in range(num_updates):
            batch = replay_buffer.sample(self.batch_size)
            x_train = torch.concat([batch.observations, batch.next_observations], 0)

            z = self.encoder(x_train)
            x_rec = self.decoder(z)
            rec_loss = (x_rec - x_train).abs().mean()

            loss = rec_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            metrics = {
                "rec_loss": rec_loss.item(),
                "loss": loss.item(),
            }

            # Update the target networks
            soft_update_params(self.encoder, self.target_encoder, tau=self.tau)
            soft_update_params(self.decoder, self.target_decoder, tau=self.tau)

            if i % 100 == 0:
                logger.info(f"Iteration {i} rec_loss {rec_loss}")
                if wandb.run is not None:
                    wandb.log(metrics)

            if self.early_stopper is not None:
                if self.early_stopper(rec_loss):
                    logger.info("Early stopping criteria met, stopping AE training...")
                    break

        self.encoder.eval()
        self.decoder.eval()
        return metrics

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()


class MLPEncoder(nn.Module):
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
        self.reset()

    def forward(self, x):
        z = self._mlp(x)
        return z

    def reset(self):
        self.apply(src.agents.utils.orthogonal_init)


class MLPDecoder(nn.Module):
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
        self.reset()

    def forward(self, z):
        x = self._mlp(z)
        return x

    def reset(self):
        self.apply(src.agents.utils.orthogonal_init)
