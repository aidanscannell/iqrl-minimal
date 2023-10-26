#!/usr/bin/env python3
import logging
from typing import List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import wandb
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from src.custom_types import ActionTrajectory, State, LatentState, Observation
from src.utils import EarlyStopper
from src.utils.buffer import ReplayBuffer
from src.world_model import WorldModel

from .agent import Agent


class Actor(nn.Module):
    def __init__(self, state_dim: int, mlp_dims: List[int], action_dim: int):
        super().__init__()
        self._actor = util.mlp(state_dim, mlp_dims, action_dim)
        self.apply(util.orthogonal_init)

    def forward(self, state: State, std):
        mu = self._actor(state)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return util.TruncatedNormal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int, mlp_dims: List[int], action_dim: int):
        super().__init__()
        self._critic1 = util.mlp(state_dim + action_dim, mlp_dims, 1)
        self._critic2 = util.mlp(state_dim + action_dim, mlp_dims, 1)
        self.apply(util.orthogonal_init)

    def forward(self, state: State, action: Action):
        state_action_input = torch.cat([state, action], dim=-1)
        return self._critic1(state_action_input), self._critic2(state_action_input)


class ALMAgent(Agent):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        world_model: WorldModel,
        batch_size: int,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        early_stopper: EarlyStopper,
        mlp_dims: List[int] = [512, 512],
        learning_rate: float = 3e-4,
        max_iterations: int = 100,  # for training encoder/trans
        # std_schedule: str = "linear(1.0, 0.1, 50)",
        std: float = 0.1,  # TODO make this schedule
        std_clip: float = 0.3,
        # nstep: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        horizon_train: int = 5,
        num_mppi_iterations: int = 5,
        num_samples: int = 512,
        mixture_coef: float = 0.05,
        num_topk: int = 64,
        temperature: int = 0.5,
        momentum: float = 0.1,
        unc_prop_strategy: str = "mean",  # "mean" or "sample", "sample" require transition_model to use SVGP prediction type
        sample_actor: bool = True,
        bootstrap: bool = True,
        wandb_loss_name: Optional[str] = None,
        device: str = "cuda",
        logging_freq: int = 500,
    ):
        self.actor = actor
        self.critic = critic
        self.world_model = world_model
        self.batch_size = batch_size
        self.early_stopper = early_stopper
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

        self.horizon_train = horizon_train

        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.latent_dim = latent_dim
        # learning_rate: float = 3e-4,
        # max_ddpg_iterations: int = 100,  # for training DDPG
        # # std_schedule: str = "linear(1.0, 0.1, 50)",
        # self.std = std
        # self.std_clip = std_clip
        self.gamma = gamma
        # self.tau = tau
        # self.horizon = horizon
        # self.num_mppi_iterations = num_mppi_iterations
        # self.num_samples = num_samples
        # self.mixture_coef = mixture_coef
        # self.num_topk = num_topk
        # self.temperature = temperature
        # self.momentum = momentum
        # self.unc_prop_strategy = unc_prop_strategy
        # self.sample_actor = sample_actor
        # self.bootstrap = bootstrap
        self.device = device = device
        self.wandb_loss_name = wandb_loss_name
        self.logging_freq = logging_freq

        # self._prev_mean = torch.zeros(horizon, action_dim, device=device)

    def train(self, replay_buffer: ReplayBuffer):
        logger.info("Starting training world model...")
        if self.early_stopper is not None:
            self.early_stopper.reset()
        # self.network.apply(weights_init_normal)
        self.world_model.train()
        encoder_opt = torch.optim.Adam(
            [{"params": self.world_model.encoder.parameters()}], lr=self.learning_rate
        )
        transition_opt = torch.optim.Adam(
            [{"params": self.world_model.transition_model.parameters()}],
            lr=self.learning_rate,
        )
        reward_opt = torch.optim.Adam(
            [{"params": self.world_model.reward_model.parameters()}],
            lr=self.learning_rate,
        )
        for i in range(self.max_iterations):
            samples = replay_buffer.sample(batch_size=self.batch_size)
            loss = -self.elbo(samples)

            encoder_opt.zero_grad()
            transition_opt.zero_grad()
            reward_opt.zero_grad()
            loss.backward()
            encoder_opt.step()
            transition_opt.step()
            reward_opt.step()

            if self.wandb_loss_name is not None:
                wandb.log({self.wandb_loss_name: loss})

            if i % self.logging_freq == 0:
                logger.info("Iteration : {} | Loss: {}".format(i, loss))
            if self.early_stopper is not None:
                stop_flag = self.early_stopper(loss)
                if stop_flag:
                    logger.info("Early stopping criteria met, stopping training")
                    logger.info("Breaking out loop")
                    break
        logger.info("Finished training world model...")

    def elbo(self, samples):
        observations = samples["state"]
        actions = samples["action"]
        next_observations = samples["next_state"]

        latent_states = self.world_model.encoder(obs=observations)
        next_latent_states_enc = self.world_model.encoder_targ(obs=next_observations)

        next_latent_states_trans, rewards = self.rollout(
            latent_state=latent_states[0], horizon=self.horizon_train
        )
        kl_loss = td.kl_divergence(next_latent_states_trans, next_latent_states_enc)

        loss = 0
        # TODO remove for loop by pre calculating gammas
        for i in range(self.horizon_train):
            loss += self.gamma**i * (rewards[i] - kl_loss[i])

        # TODO should this use action or policy?
        # TODO should this use latent from dynamics or encoder?
        # Q = self.critic(latent_states[-1], self.actor(latent_states[-1]))
        Q = self.critic(
            next_latent_states_enc[-1], self.actor(next_latent_states_enc[-1])
        )
        Q = self.critic(
            next_latent_states_trans[-1], self.actor(next_latent_states_trans[-1])
        )
        loss += self.gamma ** (i + 1) * Q
        return loss

    def rollout(self, latent_state: LatentState, horizon: int = None):
        if horizon is None:
            horizon = self.horizon_train
        next_latent_states, rewards = [], []
        for i in range(horizon):
            next_latent_state, reward = self.world_model.transition_model(
                latent_state=latent_state, action=self.actor(latent_state)
            )
            next_latent_states.append(next_latent_state)
            rewards.append(reward)
        next_latent_states = torch.stack(next_latent_states, 0)
        rewards = torch.stack(rewards, 0)
        return next_latent_states, rewards

    def _update_critic(
        self,
        latent_state: LatentState,
        action: Action,
        reward: Reward,
        discount: float,
        next_latent_state: LatentState,
    ):
        with torch.no_grad():
            action = self.actor(next_latent_state, std=self.std).sample(
                clip=self.std_clip
            )

            td_target = reward + discount * torch.min(
                *self.critic_tar(next_latent_state, action)
            )

        q1, q2 = self.critic(latent_state, act)
        q_loss = torch.mean(h.mse(q1, td_target) + h.mse(q2, td_target))

        self.critic_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_optim.step()

        return {"q": q1.mean().item(), "q_loss": q_loss.item()}

    def _update_pi(self, z):
        a = self.actor(z, std=self.std).sample(clip=self.std_clip)
        Q = torch.min(*self.critic(z, a))
        pi_loss = -Q.mean()

        self.actor_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.actor_optim.step()

        return {"pi_loss": pi_loss.item()}
