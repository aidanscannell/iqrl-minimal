#!/usr/bin/env python3
import logging
from typing import Any, List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import src
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box, Space
from src.custom_types import Action, EvalMode, State, T0
from stable_baselines3.common.buffers import ReplayBuffer

from .agent import Agent


# class Actor(nn.Module):
#     def __init__(self, obs_dim: int, mlp_dims: List[int], action_dim: int):
#         super().__init__()
#         self._actor = util.mlp(obs_dim, mlp_dims, action_dim)
#         self.apply(util.orthogonal_init)

#     def forward(self, state, std):
#         mu = self._actor(state)
#         mu = torch.tanh(mu)
#         std = torch.ones_like(mu) * std
#         return util.TruncatedNormal(mu, std)


# class Critic(nn.Module):
#     def __init__(self, obs_dim: int, mlp_dims: List[int], action_dim: int):
#         super().__init__()
#         self._critic1 = util.mlp(obs_dim + action_dim, mlp_dims, 1)
#         self._critic2 = util.mlp(obs_dim + action_dim, mlp_dims, 1)
#         self.apply(util.orthogonal_init)

#     def forward(self, state, action):
#         state_action_input = torch.cat([state, action], dim=-1)
#         return self._critic1(state_action_input), self._critic2(state_action_input)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Box,
        mlp_dims: List[int],
        act_fn=nn.ELU,
    ):
        super().__init__()
        input_dim = np.array(observation_space.shape).prod() + np.prod(
            action_space.shape
        )
        self._critic = src.agents.utils.mlp(
            in_dim=input_dim, mlp_dims=mlp_dims, out_dim=1, act_fn=act_fn
        )
        self.apply(src.agents.utils.orthogonal_init)
        # self.fc1 = nn.Linear(input_dim, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        q = self._critic(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return q


class Actor(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Box,
        mlp_dims: List[int],
        act_fn=nn.ELU,
    ):
        super().__init__()
        self._actor = src.agents.utils.mlp(
            in_dim=np.array(observation_space.shape).prod(),
            mlp_dims=mlp_dims,
            out_dim=np.prod(action_space.shape),
            act_fn=act_fn,
        )
        self.apply(src.agents.utils.orthogonal_init)
        # self.fc1 = nn.Linear(np.array(observation_space).prod(), 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc_mu = nn.Linear(256, np.prod(action_space_shape))

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

    def forward(self, x):
        x = self._actor(x)
        x = torch.tanh(x)
        return x * self.action_scale + self.action_bias
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = torch.tanh(self.fc_mu(x))
        # return x * self.action_scale + self.action_bias


class TD3(Agent):
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
        actor_update_freq: int = 2,  # update actor less frequently than critic
        # nstep: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
        name: str = "TD3",
    ):
        super().__init__(
            observation_space=observation_space, action_space=action_space, name=name
        )
        self.batch_size = batch_size
        self.num_updates = num_updates
        # self.utd_ratio = utd_ratio
        self.actor_update_freq = actor_update_freq
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.learning_rate = learning_rate
        # self.nstep = nstep
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Init actor and it's target
        self.actor = Actor(
            observation_space=observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
            act_fn=act_fn,
        ).to(device)
        self.target_actor = Actor(
            observation_space=observation_space,
            action_space=action_space,
            mlp_dims=mlp_dims,
            act_fn=act_fn,
        ).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Init two (twin) critics and their targets
        self.critic_1 = QNetwork(
            observation_space, action_space, mlp_dims=mlp_dims, act_fn=act_fn
        ).to(device)
        self.critic_2 = QNetwork(
            observation_space, action_space, mlp_dims=mlp_dims, act_fn=act_fn
        ).to(device)
        self.critic_1_target = QNetwork(
            observation_space, action_space, mlp_dims=mlp_dims, act_fn=act_fn
        ).to(device)
        self.critic_2_target = QNetwork(
            observation_space, action_space, mlp_dims=mlp_dims, act_fn=act_fn
        ).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.q_optimizer = torch.optim.AdamW(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=learning_rate,
        )
        self.actor_optimizer = torch.optim.AdamW(
            list(self.actor.parameters()), lr=learning_rate
        )

    def train(
        self,
        replay_buffer: ReplayBuffer,
        num_updates: Optional[int] = None,
        reinit_opts: bool = False,
    ) -> dict:
        if num_updates is None:
            num_updates = self.num_updates
        info = {}

        # Reinitialize the optimizers
        if reinit_opts:
            logger.info("Reinitializing actor/critic optimizers")
            self.q_optimizer = torch.optim.AdamW(
                list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                lr=self.learning_rate,
            )
            self.actor_optimizer = torch.optim.AdamW(
                list(self.actor.parameters()), lr=self.learning_rate
            )
        for i in range(num_updates):
            batch = replay_buffer.sample(self.batch_size)
            # TODO make info not overwritten at each iteration
            info.update(self.critic_update(data=batch))

            if i % self.actor_update_freq == 0:
                info.update(self.actor_update(data=batch))
        return info

    def critic_update(self, data) -> dict:
        with torch.no_grad():
            clipped_noise = (
                torch.randn_like(data.actions, device=self.device) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip) * self.target_actor.action_scale

            next_state_actions = (
                self.target_actor(data.next_observations) + clipped_noise
            ).clamp(self.action_space.low[0], self.action_space.high[0])
            critic_1_next_target = self.critic_1_target(
                data.next_observations, next_state_actions
            )
            critic_2_next_target = self.critic_2_target(
                data.next_observations, next_state_actions
            )
            min_critic_next_target = torch.min(
                critic_1_next_target, critic_2_next_target
            )
            next_q_value = data.rewards.flatten() + (
                1 - data.dones.flatten()
            ) * self.gamma * (min_critic_next_target).view(-1)

        critic_1_a_values = self.critic_1(data.observations, data.actions).view(-1)
        critic_2_a_values = self.critic_2(data.observations, data.actions).view(-1)
        critic_1_loss = F.mse_loss(critic_1_a_values, next_q_value)
        critic_2_loss = F.mse_loss(critic_2_a_values, next_q_value)
        critic_loss = critic_1_loss + critic_2_loss

        # Optimize the model
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # Update the target network
        for param, target_param in zip(
            self.critic_1.parameters(), self.critic_1_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.critic_2.parameters(), self.critic_2_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        info = {
            "critic_1_values": critic_1_a_values.mean().item(),
            "critic_2_values": critic_2_a_values.mean().item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "critic_loss": critic_loss.item() / 2.0,
        }
        return info

    def actor_update(self, data) -> dict:
        actor_loss = -self.critic_1(
            data.observations, self.actor(data.observations)
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target network
        for param, target_param in zip(
            self.actor.parameters(), self.target_actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        info = {"actor_loss": actor_loss.item()}
        return info

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        actions = self.actor(torch.Tensor(observation).to(self.device))
        if not eval_mode:
            actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
        actions = actions.cpu().numpy()
        actions = actions.clip(self.action_space.low, self.action_space.high)
        return actions
