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
from helper import EarlyStopper, soft_update_params
from utils.buffers import ReplayBuffer, ReplayBufferSamples
from vector_quantize_pytorch import FSQ, VectorQuantize


class Encoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        # act_fn=nn.ELU,
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
        # self.latent_dim = len(levels)
        self.latent_dim = (1024, len(levels))
        # self.latent_dim = (levels[0], len(levels), len(levels))
        # out_dim = self.latent_dim
        out_dim = np.array(self.latent_dim).prod()
        # print(f"out_dim {out_dim}")
        # out_dim = levels
        self._mlp = h.mlp(
            in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=None
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
        # breakpoint()
        return z, indices


class Decoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        # act_fn=nn.ELU,
    ):
        super().__init__()
        self.levels = levels
        # in_dim = np.array(levels).prod()
        # self.latent_dim = len(levels)
        self.latent_dim = (1024, len(levels))
        # self.latent_dim = (levels[0], len(levels), len(levels))
        # in_dim = self.latent_dim
        in_dim = np.array(self.latent_dim).prod()
        out_dim = np.array(observation_space.shape).prod()
        self._mlp = h.mlp(
            in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=None
        )
        self._fsq = FSQ(levels)
        self.apply(h.orthogonal_init)

    def forward(self, z):
        # print("inside decoder")
        # print(f"z {z.shape}")
        z = torch.flatten(z, -2, -1)
        # z = torch.flatten(z, -len(self.levels), -1)
        # print(f"z {z.shape}")
        x = self._mlp(z)
        # print(f"x {x.shape}")
        # c = x.clamp(-1, 1)
        # print(f"c {c.shape}")
        return x.clamp(-1, 1)
        # return x


class FSQAutoEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        # act_fn=nn.ELU,
    ):
        super().__init__()
        self.encoder = Encoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            # act_fn=act_fn,
        )
        self.decoder = Decoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            # act_fn=act_fn,
        )

    def forward(self, x):
        z, indices = self.encoder(x)
        x_rec = self.decoder(z)
        # breakpoint()
        # codes = self.encoder._fsq.get_codes_from_indices(indices)
        # print(f"codes {codes.shape}")
        return x_rec, indices, z


class VectorQuantizedDDPG(Agent):
    def __init__(
        self,
        observation_space: Space,
        action_space: Box,
        mlp_dims: List[int] = [512, 512],
        # act_fn=nn.ELU,
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
        vq_patience: int = 100,
        vq_min_delta: float = 0.0,
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        num_codes: int = 512,
        ae_tau: float = 0.005,
        use_target_encoder: bool = True,
        alpha: int = 10,
        # actor_update_freq: int = 2,  # update actor less frequently than critic
        nstep: int = 1,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
        name: str = "DDPG",
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space, action_space=action_space, name=name
        )
        self.vq_batch_size = vq_batch_size
        self.vq_learning_rate = vq_learning_rate
        self.vq_num_updates = vq_num_updates
        self.alpha = alpha
        self.num_codes = num_codes
        self.levels = levels

        self.ae_tau = ae_tau

        self.use_target_encoder = use_target_encoder

        self.device = device

        self.vq = FSQAutoEncoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            # act_fn=act_fn,
        ).to(device)
        self.vq_target = FSQAutoEncoder(
            observation_space=observation_space,
            mlp_dims=mlp_dims,
            levels=levels,
            # act_fn=act_fn,
        ).to(device)
        self.vq_target.load_state_dict(self.vq.state_dict())
        # self.vq = VectorQuantize(
        #     dim=256,
        #     codebook_size=num_codes,  # codebook size
        #     decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
        #     commitment_weight=1.0,  # the weight on the commitment loss
        # ).to(device)
        self.vq_opt = torch.optim.AdamW(self.vq.parameters(), lr=vq_learning_rate)

        self.vq_patience = vq_patience
        self.vq_min_delta = vq_min_delta

        # x = torch.randn(1, 1024, 256)
        # quantized, indices, commit_loss = self.vq(x)  # (1, 1024, 256), (1, 1024), (1)

        # TODO make a space for latent states
        # latent_observation_space = observation_space
        high = np.array(levels).prod()
        # TODO is this the right way to make observation space??
        # TODO what is highest value of latent codes?
        # TODO should shape be (levels[0], 3)??
        self.latent_observation_space = gym.spaces.Box(
            low=0.0,
            high=high,
            shape=(1024,),
            # shape=(1024, len(self.levels)),
            # shape=(512, len(self.levels)),
            # shape=(levels[0], len(self.levels), len(self.levels)),
            # shape=(levels[0], len(self.levels)),
        )
        print(f"latent_observation_space {self.latent_observation_space}")
        # Init ddpg agent
        self.ddpg = agents.DDPG(
            observation_space=self.latent_observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
            # act_fn=act_fn,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_updates=num_updates,
            discount=discount,
            tau=tau,
            device=device,
            name=name,
        )

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and DDPG at same time"""
        num_updates = int(num_new_transitions * self.ddpg.utd_ratio)
        info = {}

        logger.info(f"Performing {num_updates} DDPG-AE updates...")
        reset_flag = 0
        for i in range(num_updates):
            batch = replay_buffer.sample(self.ddpg.batch_size)
            # num_encoder_updates = 2  # 2 encoder updates for one q update
            # for i in range(num_encoder_updates):
            # Update encoder less frequently than actor/critic
            # if i % self.encoder_update_freq == 0:
            info.update(self.update_representation_step(batch=batch))

            # Map observations to latent
            if self.use_target_encoder:
                latent_obs = self.vq_target.encoder(batch.observations)[1]
                latent_next_obs = self.vq_target.encoder(batch.next_observations)[1]
            else:
                latent_obs = self.vq.encoder(batch.observations)[1]
                latent_next_obs = self.vq.encoder(batch.next_observations)[1]
            # breakpoint()
            # batch.observation = latent_obs.to(torch.float).detach()
            # batch.next_observation = latent_next_obs.to(torch.float).detach()
            latent_batch = ReplayBufferSamples(
                observations=latent_obs.to(torch.float).detach(),
                actions=batch.actions,
                next_observations=latent_next_obs.to(torch.float).detach(),
                dones=batch.dones,
                rewards=batch.rewards,
                next_state_discounts=batch.next_state_discounts,
            )

            # DDPG on latent representation
            info.update(self.ddpg.update_step(batch=latent_batch))

            # # Potentially reset ae/actor/critic NN params
            # if self.reset_strategy == "every-x-param-updates":
            #     if self.reset_params_freq is not None:
            #         if self.ddpg.critic_update_counter % self.reset_params_freq == 0:
            #             logger.info(
            #                 f"Resetting as step {self.ddpg.critic_update_counter} % {self.reset_params_freq} == 0"
            #             )
            #             self.reset(replay_buffer=replay_buffer)
            #             reset_flag = 1
            # elif self.reset_strategy == "latent-dist":
            #     if self.trigger_reset_latent_dist(replay_buffer=replay_buffer):
            #         reset_flag = 1
            # else:
            #     raise NotImplementedError(
            #         "reset_strategy should be either 'every-x-params-updates' or 'latent-dist'"
            #     )

            # if reset_flag == 1:
            #     if wandb.run is not None:
            #         wandb.log({"reset": reset_flag})

            if i % 100 == 0:
                logger.info(
                    f"Iteration {i} | loss {info['encoder_loss']} | rec loss {info['rec_loss']}"
                    # f"Iteration {i} | loss {info['encoder_loss']} | rec loss {info['rec_loss']} | tc loss {info['temporal_consitency_loss']} | reward loss {info['reward_loss']} | value dynamics loss {info['value_dynamics_loss']}"
                )
                if wandb.run is not None:
                    # info.update({"exploration_noise": self.ddpg.exploration_noise})
                    wandb.log(info)
                    # wandb.log({"reset": reset_flag})
                reset_flag = 0

        logger.info("Finished training DDPG-AE")

        return info

    def update_old(
        self,
        replay_buffer: ReplayBuffer,
        num_new_transitions: int = 1,
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
                # latent_obs = vq.encoder(batch.observations)[1].to(torch.float)
                latent_obs = vq.encoder(batch.observations)[0].to(torch.float)
                latent_obs = torch.flatten(latent_obs, -2, -1)
                # latent_obs = torch.flatten(latent_obs, -2, -1)
                # latent_next_obs = vq.encoder(batch.next_observations)[1].to(torch.float)
                latent_next_obs = vq.encoder(batch.next_observations)[0].to(torch.float)
                latent_next_obs = torch.flatten(latent_next_obs, -2, -1)
                # breakpoint()
                # latent_next_obs = torch.flatten(latent_next_obs, -2, -1)
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
            self.ddpg.update(
                replay_buffer=latent_replay_buffer,
                num_new_transitions=num_new_transitions,
            )
        )
        logger.info("Finished training DDPG")
        return info

    def update_representation_step(self, batch: ReplayBufferSamples):
        loss, info = self.representation_loss(batch=batch)

        self.vq_opt.zero_grad()
        loss.backward()
        self.vq_opt.step()

        # Update the target network
        if self.use_target_encoder:
            soft_update_params(self.vq, self.vq_target, tau=self.ae_tau)

        return info

    def representation_loss(self, batch: ReplayBufferSamples):
        x_rec, indices, z = self.vq(batch.observations)
        rec_loss = (x_rec - batch.observations).abs().mean()
        loss = rec_loss

        info = {
            "encoder_loss": loss.item(),
            "rec_loss": rec_loss.item(),
            # "cmt_loss": cmt_loss.item(),
            "active_percent": indices.unique().numel() / self.num_codes * 100,
        }
        return rec_loss, info

    def train_vq_vae(
        self, replay_buffer: ReplayBuffer, num_updates: Optional[int] = None
    ):
        if num_updates is None:
            num_updates = self.vq_num_updates
        # self.vq.encoder.apply(src.agents.utils.orthogonal_init)
        # self.vq.decoder.apply(src.agents.utils.orthogonal_init)
        # self.vq_opt = torch.optim.AdamW(self.vq.parameters(), lr=self.vq_learning_rate)
        vq_early_stopper = EarlyStopper(
            patience=self.vq_patience, min_delta=self.vq_min_delta
        )
        info = {}
        i = 0
        for _ in range(num_updates):
            batch = replay_buffer.sample(self.vq_batch_size)
            x_rec, indices, z = self.vq(batch.observations)
            rec_loss = (x_rec - batch.observations).abs().mean()

            self.vq_opt.zero_grad()
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
            i += 1
            if vq_early_stopper(rec_loss):
                logger.info("Early stopping criteria met, stopping VQ training...")
                break
            if vq_early_stopper(rec_loss):
                # logger.info("Early stopping criteria met, stopping VAE training...")
                break
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
        if observation.ndim > 2:
            if observation.shape[0] == 1:
                observation = observation[0, ...]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        observation = torch.Tensor(observation).to(self.device)
        _, indices, z = self.vq(observation)
        # print("inside selct action")
        # print(f"z {z.shape}")
        # print(f"indices {indices.shape}")
        # indices = indices.to(torch.float)
        # z = z.to(torch.float)
        indices = indices.to(torch.float)
        # indices = torch.flatten(indices, -len(self.levels), -1)
        # indices = torch.flatten(indices, -2, -1)
        # z = torch.flatten(z, -2, -1)
        # print(f"z {z.shape}")
        return self.ddpg.select_action(observation=indices, eval_mode=eval_mode, t0=t0)
        # return self.ddpg.select_action(observation=z, eval_mode=eval_mode, t0=t0)
