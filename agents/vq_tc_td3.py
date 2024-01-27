#!/usr/bin/env python3
import logging
from typing import Any, Callable, List, Optional, Tuple

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
from utils import ReplayBuffer, ReplayBufferSamples
from vector_quantize_pytorch import FSQ


logger = logging.getLogger(__name__)


class MLPResettable(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(x)

    def reset(self, reset_type: str = "last-layer"):
        if reset_type in "full":
            h.orthogonal_init(self.parameters())
        elif reset_type in "last-layer":
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])
        else:
            raise NotImplementedError


class FSQEncoder(MLPResettable):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        num_codes: int = 1024,
    ):
        in_dim = np.array(observation_space.shape).prod()
        self.levels = tuple(levels)
        self.latent_dim = (num_codes, len(levels))
        self.act_fn = None
        out_dim = np.array(self.latent_dim).prod()
        # self._mlp = h.mlp(
        #     in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=None
        # )
        mlp = h.mlp(in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=None)

        super().__init__(mlp=mlp)

        self._fsq = FSQ(levels)
        self.apply(h.orthogonal_init)

    def forward(self, x):
        # print("inside encoder")
        # print(f"x {x.shape}")
        z = self.mlp(x)
        # print(f"z {z.shape}")
        z = z.reshape((*z.shape[0:-1], *self.latent_dim))
        # print(f"z {z.shape}")
        # indices = self._fsq(z)
        if x.ndim > 2:
            z, indices = torch.func.vmap(self._fsq)(z)
        else:
            z, indices = self._fsq(z)
        # print(f"z {z.shape}")
        # print(f"indices {indices.shape}")
        # breakpoint()
        return z, indices


class FSQDecoder(MLPResettable):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        num_codes: int = 1024,
    ):
        self.levels = levels
        self.latent_dim = (num_codes, len(levels))
        in_dim = np.array(self.latent_dim).prod()
        out_dim = np.array(observation_space.shape).prod()
        mlp = h.mlp(in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=None)

        super().__init__(mlp=mlp)

        self._fsq = FSQ(levels)
        self.apply(h.orthogonal_init)

    def forward(self, z):
        # print("inside decoder")
        # print(f"z {z.shape}")
        z = torch.flatten(z, -2, -1)
        # z = torch.flatten(z, -len(self.levels), -1)
        # print(f"z {z.shape}")
        x = self.mlp(z)
        # print(f"x {x.shape}")
        # c = x.clamp(-1, 1)
        # print(f"c {c.shape}")
        return x.clamp(-1, 1)
        # return x


class FSQMLPDynamics(MLPResettable):
    def __init__(
        self,
        action_space: Space,
        mlp_dims: List[int],
        # latent_dim: int = 20,
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        num_codes: int = 1024,
    ):
        self.levels = levels
        self.latent_dim = (num_codes, len(levels))
        self.act_fn = None

        in_dim = np.array(action_space.shape).prod() + np.array(self.latent_dim).prod()
        # in_dim = np.array(self.latent_dim).prod()
        # in_dim = np.array(action_space.shape).prod() + np.array(self.latent_dim).prod()
        out_dim = np.array(self.latent_dim).prod()
        mlp = h.mlp(in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=None)

        super().__init__(mlp=mlp)

        self._fsq = FSQ(levels)
        self.reset(reset_type="full")

    def forward(self, x, a):
        if x.ndim > 2:
            x = torch.flatten(x, -2, -1)
        # breakpoint()
        x = torch.cat([x, a], 1)
        z = self.mlp(x)
        # print(f"z {z.shape}")
        z = z.reshape((*z.shape[0:-1], *self.latent_dim))
        # print(f"z {z.shape}")
        # indices = self._fsq(z)
        z, indices = self._fsq(z)
        # print(f"z {z.shape}")
        # print(f"indices {indices.shape}")
        # return indices
        return z, indices
        # z = self.mlp(x)
        # return z


class FSQMLPReward(MLPResettable):
    def __init__(
        self,
        action_space: Space,
        mlp_dims: List[int],
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        num_codes: int = 1024,
    ):
        self.levels = levels
        self.latent_dim = (num_codes, len(levels))
        self.act_fn = None

        in_dim = np.array(action_space.shape).prod() + np.array(self.latent_dim).prod()
        mlp = h.mlp(in_dim=in_dim, mlp_dims=mlp_dims, out_dim=1, act_fn=None)

        super().__init__(mlp=mlp)

        self.reset(reset_type="full")

    def forward(self, z, a):
        if z.ndim > 2:
            z = torch.flatten(z, -2, -1)
        # breakpoint()
        x = torch.cat([z, a], 1)
        r = self.mlp(x)
        return r


class Projection(MLPResettable):
    def __init__(
        self,
        mlp_dims: List[int],
        levels: List[int] = [8, 6, 5],  # target size 2^8, actual size 240
        out_dim: int = 1024,
        num_codes: int = 1024,
    ):
        self.levels = levels
        self.latent_dim = (num_codes, len(levels))
        self.act_fn = None

        in_dim = np.array(self.latent_dim).prod()
        mlp = h.mlp(in_dim=in_dim, mlp_dims=mlp_dims, out_dim=out_dim, act_fn=None)

        super().__init__(mlp=mlp)

        self.reset(reset_type="full")

    def forward(self, x):
        if x.ndim > 2:
            x = torch.flatten(x, -2, -1)
        z = self.mlp(x)
        # print(f"z {z.shape}")
        return z


class Encoder(MLPResettable):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        act_fn: Callable = None,
        # normalize: bool = True,
        # simplex_dim: int = 10,
    ):
        in_dim = np.array(observation_space.shape).prod()
        # TODO data should be normalized???
        self.latent_dim = latent_dim
        mlp = MLPResettable(
            h.mlp(
                in_dim=in_dim,
                mlp_dims=mlp_dims,
                out_dim=latent_dim,
                act_fn=act_fn,
            )
        )
        super().__init__(mlp=mlp)
        self.act_fn = act_fn

        # self.normalize = normalize
        # self.simplex_dim = simplex_dim
        self.reset(reset_type="full")

    def forward(self, x):
        z = self.mlp(x)
        return z


class Decoder(MLPResettable):
    def __init__(
        self,
        observation_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
    ):
        self.latent_dim = latent_dim
        out_dim = np.array(observation_space.shape).prod()
        mlp = MLPResettable(
            h.mlp(in_dim=latent_dim, mlp_dims=mlp_dims, out_dim=out_dim)
        )
        super().__init__(mlp=mlp)
        self.reset(reset_type="full")

    def forward(self, z):
        x = self.mlp(z)
        return x


class MLPDynamics(MLPResettable):
    def __init__(
        self,
        action_space: Space,
        mlp_dims: List[int],
        latent_dim: int = 20,
        act_fn: Callable = None
        # normalize: bool = True,
        # simplex_dim: int = 10,
    ):
        self.latent_dim = latent_dim
        in_dim = np.array(action_space.shape).prod() + latent_dim
        # if normalize:
        #     act_fn = SimNorm(dim=simplex_dim)
        # else:
        #     act_fn = None
        # TODO data should be normalized???
        mlp = MLPResettable(
            h.mlp(
                in_dim=in_dim,
                mlp_dims=mlp_dims,
                out_dim=latent_dim,
                act_fn=act_fn,
            )
        )
        super().__init__(mlp=mlp)

        self.act_fn = act_fn

        self.reset(reset_type="full")

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        z = self.mlp(x)
        return z


class MLPReward(MLPResettable):
    def __init__(self, action_space: Space, mlp_dims: List[int], latent_dim: int = 20):
        self.latent_dim = latent_dim
        in_dim = np.array(action_space.shape).prod() + latent_dim
        mlp = MLPResettable(
            h.mlp(
                in_dim=in_dim,
                mlp_dims=mlp_dims,
                out_dim=1,
                act_fn=None,  # This will use Mish
            )
        )
        super().__init__(mlp=mlp)
        self.reset(reset_type="full")

    def forward(self, z, a):
        x = torch.cat([z, a], 1)
        r = self.mlp(x)
        return r


class VQ_TC_TD3(Agent):
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
        nstep: int = 1,  # nstep used for TD returns
        horizon: int = 5,  # horizon used for representation learning
        discount: float = 0.99,
        tau: float = 0.005,
        rho: float = 0.9,  # discount for dynamics
        act_with_target: bool = False,  # if True act with target actor network
        # Reset stuff
        reset_type: str = "last_layer",  # "full" or "last-layer"
        reset_strategy: str = "latent-dist",  #  "latent-dist" or "every-x-param-updates"
        reset_params_freq: int = 100000,  # reset params after this many param updates
        reset_threshold: float = 0.01,
        retrain_after_reset: bool = True,
        reset_retrain_strategy: str = "interleaved",  # "interleaved" or "representation-first"
        max_retrain_updates: int = 5000,
        eta_ratio: float = 1.0,  # encoder-to-agent parameter update ratio (agent updates from early stopping)
        memory_size: int = 10000,
        # AE config
        train_strategy: str = "interleaved",  # "interleaved" or "representation-first"
        temporal_consistency: bool = False,  # if True include dynamic model for representation learning
        reward_loss: bool = False,  # if True include reward model for representation learning
        value_dynamics_loss: bool = False,  # if True include value prediction for representation learning
        value_enc_loss: bool = False,  # if True include value prediction for representation learning
        reconstruction_loss: bool = True,  # if True use reconstruction loss with decoder
        use_cosine_similarity_dynamics: bool = False,
        use_cosine_similarity_reward: bool = False,
        encoder_mlp_dims: List[int] = [256],
        ae_learning_rate: float = 3e-4,
        ae_batch_size: int = 128,
        # ae_num_updates: int = 1000,
        ae_utd_ratio: int = 1,  # used for representation first training
        ae_update_freq: int = 1,  # update encoder less frequently than actor/critic
        ae_patience: Optional[int] = None,
        ae_min_delta: Optional[float] = None,
        use_early_stop: bool = False,
        project_latent: bool = False,
        projection_dim: int = 1024,
        latent_dim: int = 20,
        ae_tau: float = 0.005,
        use_target_encoder: bool = False,  # if True use target encoder for actor/critic
        act_with_target_enc: bool = False,  # if True act with target encoder network
        ae_normalize: bool = True,
        simplex_dim: int = 10,
        use_fsq: bool = False,
        fsq_num_codes: int = 1024,
        fsq_levels: List[int] = [8, 6, 5],
        fsq_idx: int = 0,  # 0 uses z and 1 uses indices for actor/critic
        # encoder_reset_params_freq: int = 10000,  # reset enc params after X param updates
        compile: bool = False,
        device: str = "cuda",
        name: str = "TC_TD3",
    ):
        super().__init__(
            observation_space=observation_space, action_space=action_space, name=name
        )
        self.train_strategy = train_strategy

        self.temporal_consistency = temporal_consistency
        self.reward_loss = reward_loss
        self.value_dynamics_loss = value_dynamics_loss
        self.value_enc_loss = value_enc_loss
        self.reconstruction_loss = reconstruction_loss
        self.use_cosine_similarity_dynamics = use_cosine_similarity_dynamics
        self.use_cosine_similarity_reward = use_cosine_similarity_reward
        self.ae_learning_rate = ae_learning_rate
        self.ae_batch_size = ae_batch_size
        # self.ae_num_updates = ae_num_updates
        self.ae_utd_ratio = ae_utd_ratio
        self.encoder_update_freq = ae_update_freq

        self.early_stopper_freq = 10
        # self.ae_patience = int(ae_patience / self.early_stopper_freq)
        self.ae_patience = ae_patience
        self.ae_min_delta = ae_min_delta

        self.project_latent = project_latent
        self.projection_dim = projection_dim

        self.rho = rho

        self.latent_dim = latent_dim
        self.ae_tau = ae_tau
        self.use_target_encoder = use_target_encoder
        self.act_with_target_enc = act_with_target_enc
        self.ae_normalize = ae_normalize
        self.simplex_dim = simplex_dim
        self.use_fsq = use_fsq
        self.fsq_num_codes = fsq_num_codes
        self.fsq_levels = fsq_levels
        self.fsq_idx = fsq_idx

        self.horizon = horizon

        self.device = device

        if ae_normalize:
            act_fn = SimNorm(dim=simplex_dim)
            target_act_fn = SimNorm(dim=simplex_dim)
        else:
            act_fn = None
            target_act_fn = None
        # Init representation learning (encoder/decoder/dynamics/reward)
        if use_fsq:
            self.encoder = FSQEncoder(
                observation_space=observation_space,
                mlp_dims=encoder_mlp_dims,
                levels=fsq_levels,
                num_codes=fsq_num_codes,
            ).to(device)
            self.encoder_target = FSQEncoder(
                observation_space=observation_space,
                mlp_dims=encoder_mlp_dims,
                levels=fsq_levels,
                num_codes=fsq_num_codes,
            ).to(device)
        else:
            self.encoder = Encoder(
                observation_space=observation_space,
                mlp_dims=encoder_mlp_dims,
                latent_dim=latent_dim,
                act_fn=act_fn,
            ).to(device)
            self.encoder_target = Encoder(
                observation_space=observation_space,
                mlp_dims=encoder_mlp_dims,
                latent_dim=latent_dim,
                act_fn=target_act_fn,
            ).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        if compile:
            self.encoder = torch.compile(self.encoder, mode="default")
            self.encoder_target = torch.compile(self.encoder_target, mode="default")

        encoder_params = list(self.encoder.parameters())

        if reconstruction_loss:
            if use_fsq:
                self.decoder = FSQDecoder(
                    observation_space=observation_space,
                    mlp_dims=mlp_dims,
                    levels=fsq_levels,
                    num_codes=fsq_num_codes,
                ).to(device)
            else:
                self.decoder = Decoder(
                    observation_space=observation_space,
                    mlp_dims=mlp_dims,
                    latent_dim=latent_dim,
                ).to(device)
            if compile:
                self.decoder = torch.compile(self.decoder, mode="default")
            encoder_params += list(self.decoder.parameters())
        if temporal_consistency or value_dynamics_loss:
            if use_fsq:
                self.dynamics = FSQMLPDynamics(
                    action_space=action_space,
                    mlp_dims=mlp_dims,
                    levels=fsq_levels,
                    num_codes=fsq_num_codes,
                ).to(device)
            else:
                self.dynamics = MLPDynamics(
                    action_space=action_space,
                    mlp_dims=mlp_dims,
                    latent_dim=latent_dim,
                    act_fn=act_fn,
                ).to(device)
            if compile:
                self.dynamics = torch.compile(self.dynamics, mode="default")
            encoder_params += list(self.dynamics.parameters())
        if self.reward_loss:
            if use_fsq:
                self.reward = FSQMLPReward(
                    action_space=action_space,
                    mlp_dims=mlp_dims,
                    levels=fsq_levels,
                    num_codes=fsq_num_codes,
                ).to(device)
                if fsq_idx == 1:
                    raise NotImplementedError
            else:
                self.reward = MLPReward(
                    action_space=action_space,
                    mlp_dims=mlp_dims,
                    latent_dim=latent_dim,
                ).to(device)
            if compile:
                self.reward = torch.compile(self.reward, mode="default")
            encoder_params += list(self.reward.parameters())

        if self.project_latent:
            self.projection = Projection(
                mlp_dims=[1024],
                levels=fsq_levels,
                num_codes=fsq_num_codes,
                out_dim=projection_dim,
            ).to(device)
            if compile:
                self.projection = torch.compile(self.projection, mode="default")
            encoder_params += list(self.projection.parameters())

            self.projection_target = Projection(
                mlp_dims=[1024],
                levels=fsq_levels,
                num_codes=fsq_num_codes,
                out_dim=projection_dim,
            ).to(device)
            self.projection_target.load_state_dict(self.projection.state_dict())
            if compile:
                self.projection_target = torch.compile(
                    self.projection_target, mode="default"
                )

        self.ae_opt = torch.optim.AdamW(encoder_params, lr=ae_learning_rate)

        self.use_early_stop = use_early_stop
        if self.use_early_stop:
            if ae_patience is not None and ae_min_delta is not None:
                self.ae_early_stopper = h.EarlyStopper(
                    patience=ae_patience, min_delta=ae_min_delta
                )
            else:
                self.ae_early_stopper = None
        else:
            self.ae_early_stopper = None

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
        if use_fsq:
            high = np.array(fsq_levels).prod()

            if self.fsq_idx == 0:
                shape = (fsq_num_codes * len(fsq_levels),)
                # shape = (fsq_num_codes, len(fsq_levels))
            else:
                shape = (fsq_num_codes,)
            self.latent_observation_space = gym.spaces.Box(
                low=0, high=high, shape=shape
            )

        print(f"latent_observation_space {self.latent_observation_space}")
        # Init DDPG agent
        self.ddpg = agents.DDPG(
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
            # reset_params_freq=reset_params_freq,
            reset_params_freq=None,  # handle resetting in this class
            reset_type=None,  # handle resetting in this class
            nstep=nstep,  # N-step returns for critic training
            discount=discount,
            tau=tau,
            device=device,
            act_with_target=act_with_target,
            name=name,
            compile=compile,
        )

        self.encoder_update_conter = 0

        self.reset_type = reset_type
        self.reset_strategy = reset_strategy
        # self.encoder_reset_params_freq = encoder_reset_params_freq
        self.reset_params_freq = reset_params_freq
        self.eta_ratio = eta_ratio
        self.reset_threshold = reset_threshold
        self.retrain_after_reset = retrain_after_reset
        self.reset_retrain_strategy = reset_retrain_strategy
        self.max_retrain_updates = max_retrain_updates
        self.memory_size = memory_size

        # Init memory
        # self.old_replay_buffer = None
        self.x_mem = None
        self.z_mem = None

        self.value_weight = 1
        self.value_weight_discount = 0.99999

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        self.train()

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

        ###### Log some stuff ######
        if wandb.run is not None:
            wandb.log({"exploration_noise": self.ddpg.exploration_noise})
            wandb.log({"buffer_size": replay_buffer.size()})

        self.ddpg.exploration_noise_schedule.step()
        self.eval()
        # self.ddpg.eval()
        return info

    def update_1(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and DDPG at same time"""
        num_updates = int(num_new_transitions * self.ddpg.utd_ratio)
        info = {}

        logger.info(f"Performing {num_updates} VQ-TC-TD3 updates...")
        reset_flag = 0
        for i in range(num_updates):
            batch = replay_buffer.sample(self.ddpg.batch_size, val=False)
            # Update encoder less frequently than actor/critic
            if i % self.encoder_update_freq == 0:
                info.update(self.update_representation_step(batch=batch))

            # Map observations to latent
            # TODO don't use target here. It breaks dog?
            # TODO I used to use target here
            if self.use_target_encoder:
                z = self.encoder_target(batch.observations)[self.fsq_idx]
                z_next = self.encoder_target(batch.next_observations)[self.fsq_idx]
            else:
                z = self.encoder(batch.observations)[self.fsq_idx]
                z_next = self.encoder(batch.next_observations)[self.fsq_idx]
            if self.fsq_idx == 0:
                z = torch.flatten(z, -2, -1)
                z_next = torch.flatten(z_next, -2, -1)
            latent_batch = ReplayBufferSamples(
                observations=z.to(torch.float).detach(),
                actions=batch.actions,
                next_observations=z_next.to(torch.float).detach(),
                dones=batch.dones,
                timeouts=batch.timeouts,
                rewards=batch.rewards,
                next_state_discounts=batch.next_state_discounts,
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
                        self.reset(replay_buffer=replay_buffer)
                        reset_flag = 1
            elif self.reset_strategy == "latent-dist":
                if self.trigger_reset_latent_dist(replay_buffer=replay_buffer):
                    reset_flag = 1

            if reset_flag == 1:
                if wandb.run is not None:
                    wandb.log({"reset": reset_flag})

            if i % 100 == 0:
                logger.info(f"latent_obs {z.shape}")
                logger.info(
                    f"Iteration {i} | loss {info['encoder_loss']} | rec loss {info['rec_loss']} | tc loss {info['temporal_consitency_loss']} | reward loss {info['reward_loss']} | value dynamics loss {info['value_dynamics_loss']}"
                )
                if wandb.run is not None:
                    # info.update({"exploration_noise": self.ddpg.exploration_noise})
                    wandb.log(info)
                    wandb.log({"reset": reset_flag})
                    # z_dist = self.latent_euclidian_dist()
                    # wandb.log({"z_dist": z_dist})
                reset_flag = 0

        logger.info("Finished training DDPG-AE")

        return info

    def update_2(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and then do DDPG"""

        ###### Train the representation ######
        num_ae_updates = int(num_new_transitions * self.ae_utd_ratio)
        info = self.update_encoder(replay_buffer, num_updates=num_ae_updates)

        ###### Check for resets using distance in latent space ######
        if self.reset_strategy == "latent-dist":
            if self.trigger_reset_latent_dist(replay_buffer=replay_buffer):
                if wandb.run is not None:
                    wandb.log({"reset": 1})

        ###### Train actor/critic ######
        num_updates = int(num_new_transitions * self.ddpg.utd_ratio)
        info.update(self.update_actor_critic(replay_buffer, num_updates=num_updates))

        return info

    def update_encoder(
        self, replay_buffer: ReplayBuffer, num_updates: int, use_early_stop: bool = True
    ) -> dict:
        """Update representation and then train actor/critic"""
        if self.ae_early_stopper is not None:
            self.ae_early_stopper.reset()
            num_updates = 10000

        info = {}
        logger.info("Training AE...")
        best_val_loss, i = float("inf"), 0
        for i in range(num_updates):
            batch = replay_buffer.sample(self.ae_batch_size, val=False)
            info.update(self.update_representation_step(batch=batch))

            if i % 100 == 0:
                logger.info(
                    f"Iteration {i} | loss {info['encoder_loss']} | rec loss {info['rec_loss']} | tc loss {info['temporal_consitency_loss']} | reward loss {info['reward_loss']} | value dynamics loss {info['value_dynamics_loss']}"
                )
                if wandb.run is not None:
                    wandb.log(info)
                    # z_dist = self.latent_euclidian_dist()
                    # wandb.log({"z_dist": z_dist})
                    # wandb.log({"reset": reset_flag})
                # reset_flag = 0

            # if i % self.early_stopper_freq == 0:
            if use_early_stop:
                if self.ae_early_stopper is not None:
                    val_batch = replay_buffer.sample(self.ae_batch_size, val=True)
                    val_loss, val_info = self.representation_loss(val_batch)
                    if wandb.run is not None:
                        wandb.log({"val_encoder_loss": val_info["encoder_loss"]})
                    if self.ae_early_stopper(val_loss):
                        logger.info(
                            "Early stopping criteria met, stopping AE training..."
                        )
                        break

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        state = {
                            "encoder": self.encoder.state_dict(),
                            "encoder_target": self.encoder_target.state_dict(),
                            "ae_opt": self.ae_opt.state_dict(),
                        }
                        if self.reward_loss:
                            state.update({"reward": self.reward.state_dict()})
                        if self.reconstruction_loss:
                            state.update({"decoder": self.decoder.state_dict()})
                        if self.temporal_consistency:
                            state.update({"dynamics": self.dynamics.state_dict()})
                        if self.project_latent:
                            state.update({"projection": self.projection.state_dict()})
                            state.update(
                                {
                                    "projection_target": self.projection_target.state_dict()
                                }
                            )
                        torch.save(state, "./best_ckpt_dict.pt")
                        # logger.info("Finished saving encoder+opt best ckpt")

        if use_early_stop:
            # Load best checkpoints
            self.encoder.load_state_dict(torch.load("./best_ckpt_dict.pt")["encoder"])
            self.encoder_target.load_state_dict(
                torch.load("./best_ckpt_dict.pt")["encoder_target"]
            )
            self.ae_opt.load_state_dict(torch.load("./best_ckpt_dict.pt")["ae_opt"])
            if self.reward_loss:
                self.reward.load_state_dict(torch.load("./best_ckpt_dict.pt")["reward"])
            if self.reconstruction_loss:
                self.decoder.load_state_dict(
                    torch.load("./best_ckpt_dict.pt")["decoder"]
                )
            if self.temporal_consistency:
                self.dynamics.load_state_dict(
                    torch.load("./best_ckpt_dict.pt")["dynamics"]
                )
            if self.project_latent:
                self.projection.load_state_dict(
                    torch.load("./best_ckpt_dict.pt")["projection"]
                )
                self.projection_target.load_state_dict(
                    torch.load("./best_ckpt_dict.pt")["projection_target"]
                )

        logger.info(f"Finished training AE for {i} update steps")
        info.update({"num_ae_updates": i})
        return info

    def update_actor_critic(
        self, replay_buffer: ReplayBuffer, num_updates: int
    ) -> dict:
        """Update actor/critic"""
        # TODO This could use ddpg.update()
        logger.info(f"Performing {num_updates} actor/critic updates...")
        info = {}
        for i in range(num_updates):
            batch = replay_buffer.sample(self.ddpg.batch_size)

            ###### Map observations to latent ######
            # TODO don't use target here. It breaks dog?
            # TODO I used to use target here
            if self.use_target_encoder:
                z = self.encoder_target(batch.observations)[self.fsq_idx]
                z_next = self.encoder_target(batch.next_observations)[self.fsq_idx]
            else:
                z = self.encoder(batch.observations)[self.fsq_idx]
                z_next = self.encoder(batch.next_observations)[self.fsq_idx]
            if self.fsq_idx == 0:
                z = torch.flatten(z, -2, -1)
                z_next = torch.flatten(z_next, -2, -1)
            latent_batch = ReplayBufferSamples(
                observations=z.to(torch.float).detach(),
                actions=batch.actions,
                next_observations=z_next.to(torch.float).detach(),
                dones=batch.dones,
                timeouts=batch.timeouts,
                rewards=batch.rewards,
                next_state_discounts=batch.next_state_discounts,
            )

            ###### Train actor/critic on latent representation ######
            info.update(self.ddpg.update_step(batch=latent_batch))

            ###### Potentially reset ae/actor/critic NN params ######
            if self.reset_strategy == "every-x-param-updates":
                if self.reset_params_freq is not None:
                    if self.ddpg.critic_update_counter % self.reset_params_freq == 0:
                        logger.info(
                            f"Resetting as step {self.ddpg.critic_update_counter} % {self.reset_params_freq} == 0"
                        )
                        self.reset(replay_buffer=replay_buffer)
                        wandb.log({"reset": 1})

            if i % 100 == 0:
                if wandb.run is not None:
                    wandb.log(info)
                    wandb.log({"reset": 0})

        logger.info("Finished training actor/critic")

        return info

    @torch.compile
    def update_representation_step(self, batch: ReplayBufferSamples):
        # Reset encoder after a fixed number of updates
        self.encoder_update_conter += 1

        loss, info = self.representation_loss(batch=batch)

        self.ae_opt.zero_grad()
        loss.backward()
        self.ae_opt.step()

        # Update the target network
        soft_update_params(self.encoder, self.encoder_target, tau=self.ae_tau)
        if self.project_latent:
            soft_update_params(self.projection, self.projection_target, tau=self.ae_tau)

        return info

    def representation_loss(
        self, batch: ReplayBufferSamples
    ) -> Tuple[torch.Tensor, dict]:
        x_train = batch.observations
        if self.use_fsq:
            # if x_train.ndim > 2:
            #     z, indices = torch.func.vmap(self.encoder)(x_train)
            # else:
            z, indices = self.encoder(x_train)
        else:
            # if x_train.ndim > 2:
            #     z = torch.func.vmap(self.encoder)(x_train)
            # else:
            z = self.encoder(x_train)

        if self.reconstruction_loss:
            x_rec = self.decoder(z)
            rec_loss = (x_rec - x_train).abs().mean()
        else:
            rec_loss = torch.zeros(1).to(self.device)

        if self.temporal_consistency:
            temporal_consitency_loss, reward_loss = 0.0, 0.0
            if self.use_fsq:
                z, _ = self.encoder(batch.observations[0])
            else:
                z = self.encoder(batch.observations[0])

            dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
            timeout_or_dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
            for t in range(self.horizon):
                dones = torch.where(timeout_or_dones, dones, batch.dones[t])
                timeout_or_dones = torch.logical_or(
                    timeout_or_dones, torch.logical_or(dones, batch.timeouts[t])
                )

                # Predict next latent
                delta_z_pred = self.dynamics(z, a=batch.actions[t])
                if self.use_fsq:
                    delta_z_pred = delta_z_pred[0]
                next_z_pred = z + delta_z_pred

                # Predict next reward
                if self.reward_loss:
                    r_pred = self.reward(z=z, a=batch.actions[t])

                with torch.no_grad():
                    next_obs = batch.next_observations[t]
                    next_z_tar = self.encoder_target(next_obs)
                    if self.use_fsq:
                        next_z_tar = next_z_tar[0]
                    r_tar = batch.rewards[t]
                    assert next_obs.ndim == r_tar.ndim == 2

                if self.project_latent:
                    # next_z_pred_not_projected = next_z_pred
                    next_z_tar = self.projection_target(next_z_tar)
                    next_z_pred = self.projection(next_z_pred)

                # Don't forget this
                z = next_z_pred

                # Losses
                rho = self.rho**t
                if self.use_fsq:
                    next_z_pred = next_z_pred.flatten(-2)
                    next_z_tar = next_z_tar.flatten(-2)
                if self.use_cosine_similarity_dynamics:
                    _temporal_consitency_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)(
                        next_z_pred, next_z_tar
                    )
                else:
                    _temporal_consitency_loss = torch.mean(
                        (next_z_pred - next_z_tar) ** 2, dim=-1
                    )
                temporal_consitency_loss += rho * torch.mean(
                    (1 - timeout_or_dones.to(torch.int)) * _temporal_consitency_loss
                )
                if self.reward_loss:
                    assert r_pred.ndim == 2
                    assert r_tar.ndim == 2
                    if self.use_cosine_similarity_reward:
                        _reward_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)(
                            r_pred, r_tar
                        )
                    else:
                        _reward_loss = (r_pred[..., 0] - r_tar[..., 0]) ** 2
                    reward_loss += rho * torch.mean(
                        (1 - timeout_or_dones.to(torch.int)) * _reward_loss
                    )
                else:
                    reward_loss += torch.zeros(1).to(self.device)
        else:
            temporal_consitency_loss = torch.zeros(1).to(self.device)

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
                    z_next_enc_target = self.encoder_target(batch.next_observations)
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
            "encoder_loss": loss.item(),
            "z_min": torch.min(z).item(),
            "z_max": torch.max(z).item(),
            "z_mean": torch.mean(z.to(torch.float)).item(),
            "z_median": torch.median(z).item(),
        }
        return loss, info

    def trigger_reset_latent_dist(self, replay_buffer) -> bool:
        if self.x_mem is None:
            # if self.old_replay_buffer is None:
            logger.info("Building memory first time...")
            # logger.info("Updating memory...")
            self._update_memory(
                replay_buffer=replay_buffer, memory_size=self.memory_size
            )
            return False

        z_dist = self.latent_euclidian_dist()
        if z_dist > self.reset_threshold:
            reset = True
            logger.info(
                f"Resetting as z_dist {z_dist} > reset_threshold {self.reset_threshold}"
            )
            self.reset(replay_buffer=replay_buffer)
        else:
            reset = False
        return reset

    def _update_memory(self, replay_buffer: ReplayBuffer, memory_size: int = 10000):
        # Sample memory set from replay buffer
        if memory_size > replay_buffer.train_size():
            memory_size = replay_buffer.train_size()
        memory_batch = replay_buffer.sample(memory_size)

        # Store obs and (old) latents in class
        self.x_mem = memory_batch.observations
        with torch.no_grad():
            self.z_mem = self.encoder(self.x_mem)
            if self.use_fsq:
                # TODO do we want to use z or indices for actor/critic?
                self.z_mem = self.z_mem[self.fsq_idx]

    def latent_euclidian_dist(self) -> float:
        z_dist = 0.0
        if self.x_mem is not None:
            z_mem_pred = self.encoder(self.x_mem)
            if self.use_fsq:
                # TODO do we want to use z or indices for actor/critic?
                z_mem_pred = z_mem_pred[self.fsq_idx]
            # TODO make sure mean is over state dimensions
            # z_dist = (self.z_mem - z_mem_pred).abs().mean()
            z_dist = ((self.z_mem - z_mem_pred) ** 2).mean()
            if wandb.run is not None:
                wandb.log({"z_euclidian_distance": z_dist})
        return z_dist

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
        if self.act_with_target_enc:
            z = self.encoder_target(observation)
        else:
            z = self.encoder(observation)
        if self.use_fsq:
            # TODO do we want to use z or indices for actor/critic?
            z = z[self.fsq_idx]
        z = z.to(torch.float)

        if self.fsq_idx == 0:
            z = torch.flatten(z, -2, -1)
        action = self.ddpg.select_action(observation=z, eval_mode=eval_mode, t0=t0)
        if flag:
            action = action[None, ...]
        return action

    def reset(
        self, reset_type: Optional[str] = None, replay_buffer: ReplayBuffer = None
    ):
        if reset_type is None:
            reset_type = self.reset_type
        logger.info("Restting agent...")
        logger.info("Resetting encoder params")
        self.encoder.reset(reset_type=reset_type)
        encoder_params = list(self.encoder.parameters())
        if self.reconstruction_loss:
            logger.info("Resetting decoder")
            self.decoder.reset(reset_type=reset_type)
            encoder_params += list(self.decoder.parameters())
        if self.temporal_consistency or self.value_dynamics_loss:
            logger.info("Resetting dynamics")
            self.dynamics.reset(reset_type=reset_type)
            encoder_params += list(self.dynamics.parameters())
        if self.reward_loss:
            logger.info("Resetting reward")
            self.reward.reset(reset_type=reset_type)
            encoder_params += list(self.reward.parameters())
        # self.ae_target.reset(full_reset=full_reset)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.ae_opt = torch.optim.AdamW(encoder_params, lr=self.ae_learning_rate)

        logger.info("Resetting actor/critic")
        self.ddpg.reset(reset_type=reset_type)

        # Don't reset during retraining
        reset_strategy = self.reset_strategy
        self.reset_strategy = None

        # Calculate number of updates to perform
        max_new_data = self.max_retrain_updates / self.ddpg.utd_ratio
        num_new_transitions = np.min([replay_buffer.size(), max_new_data])

        if self.reset_retrain_strategy == "interleaved":
            logger.info(f"Retraining interleaved...")
            # Set large num encoder updates as will be stopped early using val loss
            info = self.update_1(
                replay_buffer=replay_buffer,
                num_new_transitions=num_new_transitions,
            )
        elif self.reset_retrain_strategy == "representation-first":
            logger.info(f"Retraining representation-first...")
            # self.retrain_utd_ratio = self.ae_utd_ratio
            # Set large num encoder updates as will be stopped early using val loss
            # num_new_transitions = replay_buffer.size() * self.retrain_utd_ratio
            # num_new_transitions = replay_buffer.size() * self.ae_utd_ratio
            max_new_data = self.max_retrain_updates / self.ddpg.utd_ratio
            num_new_transitions = np.min([replay_buffer.size(), max_new_data])
            info = self.update_2(
                replay_buffer=replay_buffer, num_new_transitions=num_new_transitions
            )
        elif self.reset_retrain_strategy == "representation-only":
            logger.info(f"Retraining representation-only...")
            self.retrain_utd_ratio = self.ae_utd_ratio
            # Set large num encoder updates as will be stopped early using val loss
            num_ae_updates = replay_buffer.size() * self.retrain_utd_ratio
            logger.info(f"Retraining encoder...")
            info = self.update_encoder(
                replay_buffer,
                num_updates=num_ae_updates,
                use_early_stop=self.use_early_stop,
            )
            logger.info(f"Finished retraining encoder")
        else:
            logger.info("Not retraining after reset")
        self.reset_strategy = reset_strategy

        ###### Build memory for reset strategy ######
        if self.reset_strategy == "latent-dist":
            logger.info("Updating memory...")
            self._update_memory(
                replay_buffer=replay_buffer, memory_size=self.memory_size
            )

        # if self.retrain_after_reset:
        #     if replay_buffer is not None:
        #         # Set large num encoder updates as will be stopped early using val loss
        #         num_ae_updates = replay_buffer.size() * self.ae_utd_ratio

        #         # Train the representation
        #         logger.info(f"Finished resetting encoder/actor/critic")
        #         logger.info(f"Retraining encoder...")
        #         info = self.update_encoder(replay_buffer, num_updates=num_ae_updates)
        #         logger.info(f"Finished retraining encoder")

        #         ###### Build memory for reset strategy ######
        #         if self.reset_strategy == "latent-dist":
        #             logger.info("Updating memory...")
        #             self._update_memory(
        #                 replay_buffer=replay_buffer, memory_size=self.memory_size
        #             )

        #         # Train actor/critic
        #         logger.info(f"Retraining actor/critic...")
        #         # TODO calculate number of actor/critic updates from num_ae_updates
        #         num_ddpg_updates = int(info["num_ae_updates"] * self.eta_ratio)
        #         info = self.update_actor_critic(
        #             replay_buffer, num_updates=num_ddpg_updates
        #         )

    def train(self):
        self.encoder.train()
        self.ddpg.train()
        if self.reconstruction_loss:
            self.decoder.train()
        if self.temporal_consistency or self.value_dynamics_loss:
            self.dynamics.train()
        if self.reward_loss:
            self.reward.train()

    def eval(self):
        self.encoder.eval()
        self.ddpg.eval()
        if self.reconstruction_loss:
            self.decoder.eval()
        if self.temporal_consistency or self.value_dynamics_loss:
            self.dynamics.eval()
        if self.reward_loss:
            self.reward.eval()
