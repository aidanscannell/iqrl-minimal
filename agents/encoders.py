#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from typing import List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import helper as h
import numpy as np
import torch
import torch.nn as nn
import wandb
from custom_types import BatchLatent, BatchObservation
from gymnasium.spaces import Space
from helper import EarlyStopper, soft_update_params
from stable_baselines3.common.buffers import ReplayBuffer
from vector_quantize_pytorch import FSQ, VectorQuantize


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

    # def trigger_reset(self):
    #     z_mem_pred = self(self.x_mem)
    #     mem_dist = (self.z_mem - z_mem_pred).abs().mean()
    #     if mem_dist > threshold:
    #         reset = True
    #     else:
    #         reset = False
    #     return reset

    # def _update_memory(self, replay_buffer: ReplayBuffer, memory_size: int = 10000):
    #     # self.use_memory = True
    #     # if self.use_memory:
    #     #     self.use_memory_flag = True
    #     # memory_size = replay_buffer.size()

    #     memory = replay_buffer.sample(memory_size)
    #     # print(f"memory {memory}")
    #     # self.x_mem = torch.concat([memory.observations, memory.next_observations], 0)
    #     self.x_mem = memory.observations
    #     print(f"self.x_mem {self.x_mem.shape}")
    #     with torch.no_grad():
    #         self.z_mem = self(self.x_mem)
    #     print(f"self.z_mem {self.z_mem.shape}")

    # def __post_init__(self):
    #     super().__init__()
    def reset(self, full_reset: bool = False):
        raise NotImplementedError


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
        # encoder_reset_params_freq: int = 10000,  # reset enc params after X param updates
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
        # self.encoder_reset_params_freq = encoder_reset_params_freq
        self.early_stopper = early_stopper
        self.device = device
        self.name = name

        self.encoder = MLPEncoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        ).to(device)
        self.target_encoder = MLPEncoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        ).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.decoder = MLPDecoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        ).to(device)
        self.target_decoder = MLPDecoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            latent_dim=latent_dim,
            act_fn=act_fn,
        ).to(device)
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

    def reset(self, full_reset: bool = False):
        logger.info("Resetting encoder/decoder params")
        self.encoder.reset(full_reset=full_reset)
        self.decoder.reset(full_reset=full_reset)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_decoder.load_state_dict(self.decoder.state_dict())
        self.opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
        )


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
        self._mlp = h.mlp(
            in_dim=in_dim, mlp_dims=mlp_dims, out_dim=latent_dim, act_fn=act_fn
        )
        self.reset(full_reset=True)

    def forward(self, x):
        z = self._mlp(x)
        return z

    def reset(self, full_reset: bool = False):
        if full_reset:
            self.apply(h.orthogonal_init)
        else:
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])


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


class FSQAutoEncoder(Encoder):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        act_fn=nn.ELU,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        utd_ratio: int = 1,
        tau: float = 0.005,
        # encoder_reset_params_freq: int = 10000,  # reset enc params after X param updates
        early_stopper: Optional[EarlyStopper] = None,
        # early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)
        device: str = "cuda",
        name: str = "AE",
    ):
        latent_dim = (levels[0], len(levels), len(levels))
        latent_dim = np.array(latent_dim).prod()
        super().__init__(latent_dim=latent_dim)
        self.observation_space = observation_space
        self.mlp_dims = mlp_dims
        self.levels = levels
        self.act_fn = act_fn

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.tau = tau
        # self.encoder_reset_params_freq = encoder_reset_params_freq
        self.early_stopper = early_stopper
        self.device = device
        self.name = name

        class FSQEncoder(nn.Module):
            def __init__(
                self,
                observation_space: Space,
                mlp_dims: List[int],
                levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
                act_fn=nn.ELU,
            ):
                super().__init__()
                in_dim = np.array(observation_space.shape).prod()
                self.levels = tuple(levels)
                # TODO is this the right way to make latent dim???
                # TODO data should be normalized???
                self.latent_dim = (levels[0], len(levels), len(levels))
                out_dim = np.array(self.latent_dim).prod()
                # print(f"out_dim {out_dim}")
                # out_dim = levels
                self._mlp = h.mlp(
                    in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=act_fn
                )
                self._fsq = FSQ(levels)
                self.apply(h.orthogonal_init)

            def forward(self, x):
                # print("inside encoder")
                # print(f"x {x.shape}")
                z = self._mlp(x)
                # print(f"z {z.shape}")
                z = z.reshape((*z.shape[0:-1], *self.latent_dim))
                # print(f"z {z.shape}")
                # indices = self._fsq(z)
                z, indices = self._fsq(z)
                # print(f"z {z.shape}")
                # print(f"indices {indices.shape}")
                return z, indices

            def reset(self, full_reset: bool = False):
                if full_reset:
                    self.apply(h.orthogonal_init)
                else:
                    params = list(self.parameters())
                    h.orthogonal_init(params[-2:])

        class FSQDecoder(nn.Module):
            def __init__(
                self,
                observation_space: Space,
                mlp_dims: List[int],
                levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
                act_fn=nn.ELU,
            ):
                super().__init__()
                self.levels = levels
                # in_dim = np.array(levels).prod()
                self.latent_dim = (levels[0], len(levels), len(levels))
                in_dim = np.array(self.latent_dim).prod()
                out_dim = np.array(observation_space.shape).prod()
                self._mlp = h.mlp(
                    in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=act_fn
                )
                self._fsq = FSQ(levels)
                self.apply(h.orthogonal_init)

            def forward(self, z):
                # print("inside decoder")
                # print(f"z {z.shape}")
                z = torch.flatten(z, -len(self.levels), -1)
                # print(f"z {z.shape}")
                x = self._mlp(z)
                # print(f"x {x.shape}")
                # c = x.clamp(-1, 1)
                # print(f"c {c.shape}")
                return x.clamp(-1, 1)
                # return x

            def reset(self, full_reset: bool = False):
                if full_reset:
                    self.apply(h.orthogonal_init)
                else:
                    params = list(self.parameters())
                    h.orthogonal_init(params[-2:])

        self.encoder = FSQEncoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            act_fn=act_fn,
        ).to(device)
        self.target_encoder = FSQEncoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            act_fn=act_fn,
        ).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.decoder = FSQDecoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            act_fn=act_fn,
        ).to(device)
        self.target_decoder = FSQDecoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            act_fn=act_fn,
        ).to(device)
        self.target_decoder.load_state_dict(self.decoder.state_dict())

        self.opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )

    def __call__(
        self, observation: BatchObservation, target: bool = False
    ) -> BatchLatent:
        if target:
            z, indices = self.target_encoder(observation)
        else:
            z, indices = self.encoder(observation)

        z = torch.flatten(z, -3, -1)
        return z

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int):
        if self.early_stopper is not None:
            self.early_stopper.reset()

        self.encoder.train()
        self.decoder.train()
        num_updates = num_new_transitions * self.utd_ratio

        for i in range(num_updates):
            batch = replay_buffer.sample(self.batch_size)
            # x_train = torch.concat([batch.observations, batch.next_observations], 0)
            x_train = batch.observations

            z = self.encoder(x_train)
            z, indices = self.encoder(x_train)
            x_rec = self.decoder(z)
            rec_loss = (x_rec - x_train).abs().mean()

            loss = rec_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            metrics = {
                "rec_loss": rec_loss.item(),
                "loss": loss.item(),
                "active_percent": indices.unique().numel() / self.latent_dim * 100,
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
                    logger.info("Early stopping criteria met, stopping VQ training...")
                    break

        self.encoder.eval()
        self.decoder.eval()
        return metrics

    def reset(self, full_reset: bool = False):
        logger.info("Resetting encoder/decoder params")
        self.encoder.reset(full_reset=full_reset)
        self.decoder.reset(full_reset=full_reset)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_decoder.load_state_dict(self.decoder.state_dict())
        self.opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
        )
