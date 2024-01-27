#!/usr/bin/env python3
import logging
from typing import List, Optional, Tuple

import helper as h
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from custom_types import Agent, BatchAction, BatchObservation, BatchValue, EvalMode, T0
from gymnasium.spaces import Box, Space
from helper import soft_update_params
from utils import ReplayBuffer, ReplayBufferSamples


logger = logging.getLogger(__name__)


class Critic(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Box,
        mlp_dims: List[int],
    ):
        super().__init__()
        input_dim = np.array(observation_space.shape).prod() + np.prod(
            action_space.shape
        )
        self._q1 = h.mlp(in_dim=input_dim, mlp_dims=mlp_dims, out_dim=1)
        self._q2 = h.mlp(in_dim=input_dim, mlp_dims=mlp_dims, out_dim=1)
        self.reset(reset_type="full")

    def forward(
        self, observation: BatchObservation, action: BatchAction
    ) -> Tuple[BatchValue, BatchValue]:
        x = torch.cat([observation, action], -1)
        q1 = self._q1(x)
        q2 = self._q2(x)
        return q1, q2

    def reset(self, reset_type: str = "last-layer"):
        for q in [self._q1, self._q2]:
            if reset_type in "full":
                h.orthogonal_init(q.parameters())
                # q.apply(h.orthogonal_init)
            elif reset_type in "last-layer":
                params = list(q.parameters())
                h.orthogonal_init(params[-2:])
            else:
                raise NotImplementedError


class Actor(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Box,
        mlp_dims: List[int],
    ):
        super().__init__()
        self._actor = h.mlp(
            in_dim=np.array(observation_space.shape).prod(),
            mlp_dims=mlp_dims,
            out_dim=np.prod(action_space.shape),
        )
        self.reset(reset_type="full")

        # Action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, observation: BatchObservation) -> BatchAction:
        x = self._actor(observation)
        x = torch.tanh(x)
        action = x * self.action_scale + self.action_bias
        return action
        # return util.TruncatedNormal(x, std)

    def reset(self, reset_type: str = "last-layer"):
        if reset_type in "full":
            h.orthogonal_init(self.parameters())
            # self.apply(h.orthogonal_init)
        elif reset_type in "last-layer":
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])
        else:
            raise NotImplementedError


class DDPG(Agent):
    def __init__(
        self,
        observation_space: Space,
        action_space: Box,
        mlp_dims: List[int] = [512, 512],
        exploration_noise_start: float = 1.0,
        exploration_noise_end: float = 0.1,
        exploration_noise_num_steps: int = 50,  # number of episodes do decay noise
        # exploration_noise: float = 0.2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        utd_ratio: int = 1,  # parameter update-to-data ratio
        actor_update_freq: int = 1,  # update actor less frequently than critic
        reset_params_freq: Optional[
            int
        ] = None,  # reset params after this many param updates
        reset_type: str = "last_layer",  #  "full" or "last-layer"
        # nstep: int = 1,
        discount: float = 0.99,
        tau: float = 0.005,
        act_with_target: bool = False,  # if True act with target network
        device: str = "cuda",
        name: str = "DDPG",
        compile: bool = False,
        nstep: int = 1,
        **kwargs,  # hack to let work with agent.latent_dim in env config
    ):
        super().__init__(
            observation_space=observation_space, action_space=action_space, name=name
        )
        self.mlp_dims = mlp_dims
        self.exploration_noise_schedule = h.LinearSchedule(
            start=exploration_noise_start,
            end=exploration_noise_end,
            num_steps=exploration_noise_num_steps,
        )
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.actor_update_freq = actor_update_freq  # Should be 1 for true DDPG
        self.reset_params_freq = reset_params_freq
        self.reset_type = reset_type
        self.nstep = 1
        self.discount = discount
        self.tau = tau
        self.act_with_target = act_with_target
        self.device = device

        # Init actor and it's target
        self.actor = Actor(
            observation_space=observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
        ).to(device)
        self.target_actor = Actor(
            observation_space=observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
        ).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Init critic and it's targets
        self.critic = Critic(observation_space, action_space, mlp_dims=mlp_dims).to(
            device
        )
        self.target_critic = Critic(
            observation_space, action_space, mlp_dims=mlp_dims
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        if compile:
            self.actor = torch.compile(self.actor, mode="default")
            self.target_actor = torch.compile(self.target_actor, mode="default")
            self.critic = torch.compile(self.critic, mode="default")
            self.target_critic = torch.compile(self.target_critic, mode="default")

        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            list(self.critic.parameters()), lr=learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()), lr=learning_rate
        )

        # Counters for number of param updates
        self.critic_update_counter = 0
        self.actor_update_counter = 0

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        num_updates = int(num_new_transitions * self.utd_ratio)
        logger.info(f"Performing {num_updates} DDPG updates")

        if wandb.run is not None:
            wandb.log({"exploration_noise": self.exploration_noise})

        reset_flag = 0
        for i in range(num_updates):
            batch = replay_buffer.sample(self.batch_size)

            info = self.update_step(batch=batch)

            # Reset actor/critic after a fixed number of parameter updates
            if self.trigger_reset():
                reset_flag = 1
                if wandb.run is not None:
                    wandb.log({"reset": reset_flag})

            if i % 100 == 0:
                if wandb.run is not None:
                    wandb.log(info)
                    wandb.log({"reset": reset_flag})
                reset_flag = 0

        self.exploration_noise_schedule.step()

        return info

    @torch.compile
    def update_step(self, batch: ReplayBufferSamples) -> dict:
        info = {}

        # Update critic
        info.update(self.critic_update_step(data=batch))

        # Update actor less frequently than critic
        if self.critic_update_counter % self.actor_update_freq == 0:
            info.update(self.actor_update_step(data=batch))

        return info

    def critic_update_step(self, data: ReplayBufferSamples) -> dict:
        self.critic_update_counter += 1

        with torch.no_grad():
            clipped_noise = (
                torch.randn_like(data.actions, device=self.device) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip) * self.target_actor.action_scale

            next_state_actions = (
                self.target_actor(data.next_observations) + clipped_noise
            ).clamp(self.action_space.low[0], self.action_space.high[0])
            q1_next_target, q2_next_target = self.target_critic(
                data.next_observations, next_state_actions
            )
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            next_q_value = data.rewards.flatten() + (
                1 - data.dones.flatten()
            ) * data.next_state_discounts.flatten() * (min_q_next_target).view(-1)
        #            ) * self.discount**self.nstep * (min_q_next_target).view(-1)

        q1_values, q2_values = self.critic(data.observations, data.actions)
        q1_loss = F.mse_loss(q1_values.view(-1), next_q_value)
        q2_loss = F.mse_loss(q2_values.view(-1), next_q_value)
        q_loss = q1_loss + q2_loss

        # Optimize the model
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update the target network
        soft_update_params(self.critic, self.target_critic, tau=self.tau)

        info = {
            # "q1_values": q1_values.mean().item(),
            # "q2_values": q2_values.mean().item(),
            # "q1_loss": q1_loss.item(),
            # "q2_loss": q2_loss.item(),
            "q_loss": q_loss.item() / 2,
            "critic_update_counter": self.critic_update_counter,
        }
        return info

    def actor_update_step(self, data: ReplayBufferSamples) -> dict:
        self.actor_update_counter += 1
        q1, q2 = self.critic(data.observations, self.actor(data.observations))
        actor_loss = -torch.min(q1, q2).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target network
        soft_update_params(self.actor, self.target_actor, tau=self.tau)

        info = {
            "actor_loss": actor_loss.item(),
            "actor_update_counter": self.actor_update_counter,
        }
        return info

    def trigger_reset(self) -> bool:
        """Returns True if it has reset and False otherwise"""
        reset = False
        if self.reset_params_freq is not None:
            if self.critic_update_counter % self.reset_params_freq == 0:
                logger.info(
                    f"Resetting as step {self.critic_update_counter} % {self.reset_params_freq} == 0"
                )
                self.reset(reset_type=self.reset_type)
                reset = True
        return reset

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        if self.act_with_target:
            actions = self.target_actor(torch.Tensor(observation).to(self.device))
        else:
            actions = self.actor(torch.Tensor(observation).to(self.device))
        if not eval_mode:
            actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
        actions = actions.cpu().numpy()
        actions = actions.clip(self.action_space.low, self.action_space.high)
        return actions

    @property
    def exploration_noise(self):
        return self.exploration_noise_schedule()

    def reset(self, reset_type: str = "last-layer"):
        logger.info("Resetting actor/critic params")
        self.actor.reset(reset_type=reset_type)
        self.critic.reset(reset_type=reset_type)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=self.learning_rate
        )
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.learning_rate
        )
        # TODO more updates after resetting?
        # for j in range(replay_buffer.size() - num_updates):
        #     batch = replay_buffer.sample(self.batch_size)
        #     info = self.update_step(batch=batch, i=i + j)
        # self.critic_update_counter = 1
        # self.actor_update_counter = 1
        # break

        # self.critic_update_counter += 1
