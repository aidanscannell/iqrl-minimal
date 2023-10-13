#!/usr/bin/env python3
import logging
from typing import Any, List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import wandb
import gymnasium as gym
import numpy as np
import src
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box, Space
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
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        act_fn=nn.ELU,
    ):
        super().__init__()
        in_dim = np.array(observation_space.shape).prod()
        # print(f"encoder in_dim {in_dim}")
        self.levels = tuple(levels)
        # print(f"levels {levels}")
        # out_dim = tuple(levels)
        # out_dim = np.array(levels).prod()
        # out_dim = np.array((8, 3, 3)).prod()
        # TODO is this the right way to make latent dim???
        # TODO data should be normalized???
        self.latent_dim = (levels[0], len(levels), len(levels))
        out_dim = np.array(self.latent_dim).prod()
        # print(f"out_dim {out_dim}")
        # out_dim = levels
        self._mlp = src.agents.utils.mlp(
            in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=act_fn
        )
        self._fsq = FSQ(levels)
        self.apply(src.agents.utils.orthogonal_init)

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


class Decoder(nn.Module):
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
        self._mlp = src.agents.utils.mlp(
            in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=act_fn
        )
        self._fsq = FSQ(levels)
        self.apply(src.agents.utils.orthogonal_init)

    def forward(self, z):
        # print("inside decoder")
        # print(f"z {z.shape}")
        z = torch.flatten(z, -len(self.levels), -1)
        # print(f"z {z.shape}")
        x = self._mlp(z)
        # print(f"x {x.shape}")
        # c = x.clamp(-1, 1)
        # print(f"c {c.shape}")
        # return x.clamp(-1, 1)
        return x


class FSQAutoEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        act_fn=nn.ELU,
    ):
        super().__init__()
        self.encoder = Encoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            act_fn=act_fn,
        )
        self.decoder = Decoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            act_fn=act_fn,
        )

    def forward(self, x):
        z, indices = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec.clamp(-1, 1), indices


class VectorQuantizedDDPG(Agent):
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
        # VQ config
        vq_learning_rate: float = 3e-4,
        vq_batch_size: int = 128,
        vq_num_updates: int = 1000,
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        num_codes: int = 512,
        alpha: int = 10,
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
        self.vq_batch_size = vq_batch_size
        self.vq_learning_rate = vq_learning_rate
        self.vq_num_updates = vq_num_updates
        self.alpha = alpha
        self.num_codes = num_codes
        self.levels = levels

        self.device = device

        self.vq = FSQAutoEncoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            act_fn=act_fn,
        ).to(device)
        # self.vq = VectorQuantize(
        #     dim=256,
        #     codebook_size=num_codes,  # codebook size
        #     decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
        #     commitment_weight=1.0,  # the weight on the commitment loss
        # ).to(device)
        self.vq_opt = torch.optim.AdamW(self.vq.parameters(), lr=vq_learning_rate)

        # x = torch.randn(1, 1024, 256)
        # quantized, indices, commit_loss = self.vq(x)  # (1, 1024, 256), (1, 1024), (1)

        # TODO make a space for latent states
        # latent_observation_space = observation_space
        high = np.array(levels).prod()
        # TODO is this the right way to make observation space??
        # TODO what is highest value of latent codes?
        # TODO should shape be (levels[0], 3)??
        self.latent_observation_space = gym.spaces.Box(
            low=0.0, high=high, shape=(levels[0], len(self.levels))
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
        logger.info("Training VQ-VAE...")
        info = self.train_vq_vae(replay_buffer=replay_buffer)
        # info = self.train_vq_vae(replay_buffer=replay_buffer, num_updates=num_updates)
        logger.info("Finished training VQ-VAE")

        vq = self.vq

        class LatentReplayBuffer:
            def sample(self, batch_size: int):
                batch = replay_buffer.sample(batch_size=batch_size)
                latent_obs = vq.encoder(batch.observations)[1].to(torch.float)
                latent_obs = torch.flatten(latent_obs, -2, -1)
                latent_next_obs = vq.encoder(batch.next_observations)[1].to(torch.float)
                latent_next_obs = torch.flatten(latent_next_obs, -2, -1)
                batch = ReplayBufferSamples(
                    observations=latent_obs.detach(),
                    actions=batch.actions,
                    next_observations=latent_next_obs.detach(),
                    dones=batch.dones,
                    rewards=batch.rewards,
                )
                return batch

        latent_replay_buffer = LatentReplayBuffer()
        logger.info("Training DDPG...")
        info.update(
            self.ddpg.train(replay_buffer=latent_replay_buffer, num_updates=num_updates)
        )
        logger.info("Finished training DDPG")
        return info

    def train_vq_vae(
        self, replay_buffer: ReplayBuffer, num_updates: Optional[int] = None
    ):
        if num_updates is None:
            num_updates = self.vq_num_updates
        # self.vq.encoder.apply(src.agents.utils.orthogonal_init)
        # self.vq.decoder.apply(src.agents.utils.orthogonal_init)
        # self.vq_opt = torch.optim.AdamW(self.vq.parameters(), lr=self.vq_learning_rate)
        info = {}
        i = 0
        for _ in range(num_updates):
            self.vq_opt.zero_grad()
            batch = replay_buffer.sample(self.vq_batch_size)

            for x_train in (batch.observations, batch.next_observations):
                x_rec, indices = self.vq(x_train)
                rec_loss = (x_rec - x_train).abs().mean()
                i += 1
                rec_loss.backward()

                self.vq_opt.step()

                if i % 100 == 0:
                    print(f"Iteration {i} rec_loss {rec_loss}")
                    try:
                        wandb.log(
                            {
                                "rec_loss": rec_loss.item(),
                                # "cmt_loss": cmt_loss.item(),
                                "active_percent": indices.unique().numel()
                                / self.num_codes
                                * 100,
                            }
                        )
                    except:
                        pass
                # info.update(
                #     {
                #         "rec_loss": rec_loss.item(),
                #         # "cmt_loss": cmt_loss.item(),
                #         "active_percent": indices.unique().numel() / self.num_codes * 100,
                #     }
                # )
                # pbar.set_description(
                #     f"rec loss: {rec_loss.item():.3f} | "
                #     + f"cmt loss: {cmt_loss.item():.3f} | "
                #     + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
                # )
        return info

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        observation = torch.Tensor(observation).to(self.device)
        _, indices = self.vq(observation)
        indices = indices.to(torch.float)
        # indices = torch.flatten(indices, -len(self.levels), -1)
        indices = torch.flatten(indices, -2, -1)
        return self.ddpg.select_action(observation=indices, eval_mode=eval_mode, t0=t0)
