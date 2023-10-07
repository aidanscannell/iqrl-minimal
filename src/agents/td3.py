#!/usr/bin/env python3
import logging
import time
from typing import List, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import src.agents.utils as util
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
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
    def __init__(self, observation_space_shape, action_space_shape):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(observation_space_shape).prod() + np.prod(action_space_shape), 256
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(
        self,
        observation_space_shape,
        action_space_shape,
        action_space_low: float,
        action_space_high: float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space_shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(action_space_shape))

        # Action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space_high - action_space_low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space_high + action_space_low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class TD3(Agent):
    def __init__(
        self,
        observation_space_shape: Tuple,
        action_space_shape: Tuple,
        action_space_low: np.ndarray,
        action_space_high: np.ndarray,
        mlp_dims: List[int] = [512, 512],
        exploration_noise: float = 0.2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        max_iterations: int = 100,
        # std: float = 0.1,  # TODO make this schedule
        # std_clip: float = 0.3,
        # nstep: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        # use_wandb: bool = True,
        device: str = "cuda",
    ):
        self.observation_space_shape = observation_space_shape
        self.action_space_shape = action_space_shape
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        # self.std = std
        # self.std_clip = std_clip
        # self.nstep = nstep
        self.gamma = gamma
        self.tau = tau
        # self.use_wandb = use_wandb
        self.device = device

        # Init actor and it's target
        self.actor = Actor(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            action_space_low=action_space_low,
            action_space_high=action_space_high,
        ).to(device)
        self.target_actor = Actor(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            action_space_low=action_space_low,
            action_space_high=action_space_high,
        ).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Init two (twin) critics and their targets
        self.critic_1 = QNetwork(observation_space_shape, action_space_shape).to(device)
        self.critic_2 = QNetwork(observation_space_shape, action_space_shape).to(device)
        self.critic_1_target = QNetwork(observation_space_shape, action_space_shape).to(
            device
        )
        self.critic_2_target = QNetwork(observation_space_shape, action_space_shape).to(
            device
        )
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=learning_rate,
        )
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()), lr=learning_rate
        )

    def train(self, replay_buffer: ReplayBuffer, global_step: int) -> dict:
        info = {"global_step": global_step}
        for i in range(self.max_iterations):
            batch = replay_buffer.sample(self.batch_size)
            # TODO make info not overwritten at each iteration
            info.update(self.critic_update(data=batch))
            # if global_step % self.policy_frequency == 0:
            info.update(self.actor_update(data=batch))
        return info

    def critic_update(self, data) -> dict:
        with torch.no_grad():
            clipped_noise = (
                torch.randn_like(data.actions, device=self.device) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip) * self.target_actor.action_scale

            next_state_actions = (
                self.target_actor(data.next_observations) + clipped_noise
            ).clamp(self.action_space_low[0], self.action_space_high[0])
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

        # if global_step % self.logging_epoch_freq == 0:
        #     # if self.writer is not None:
        #     self.writer.add_scalar(
        #         "losses/critic_1_values", critic_1_a_values.mean().item(), global_step
        #     )
        #     self.writer.add_scalar(
        #         "losses/critic_2_values", critic_2_a_values.mean().item(), global_step
        #     )
        #     self.writer.add_scalar(
        #         "losses/critic_1_loss", critic_1_loss.item(), global_step
        #     )
        #     self.writer.add_scalar(
        #         "losses/critic_2_loss", critic_2_loss.item(), global_step
        #     )
        #     self.writer.add_scalar(
        #         "losses/critic_loss", critic_loss.item() / 2.0, global_step
        #     )
        info = {
            "critic_1_values": critic_1_a_values.mean().item(),
            "critic_2_values": critic_2_a_values.mean().item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "critic_loss": critic_loss.item() / 2.0,
        }
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # self.writer.add_scalar(
        #     "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        # )
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

    # def update(self, states, actions, rewards, next_states) -> dict:
    #     info = self._update_q_fn(
    #         state=states,
    #         action=actions,
    #         reward=rewards,
    #         discount=1,  # TODO should discount be 1?
    #         next_state=next_states,
    #         std=self.std,
    #     )
    #     info.update(self._update_actor_fn(state=states, std=self.std))

    #     # Update target network
    #     util.soft_update_params(self.critic, self.critic_target, tau=self.tau)
    #     wandb.log(info)
    #     return info

    @torch.no_grad()
    def select_action(self, observation, eval_mode: EvalMode = False, t0: T0 = None):
        actions = self.actor(torch.Tensor(observation).to(self.device))
        if not eval_mode:
            actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
        actions = (
            actions.cpu().numpy().clip(self.action_space_low, self.action_space_high)
        )
        return actions
        # if isinstance(state, np.ndarray):
        #     state = torch.from_numpy(state).to(self.device).float()
        # if state.ndim == 2:
        #     # TODO added this but not checked its right
        #     # TODO previously had torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #     assert state.shape[0] == 1
        #     state = state[0, ...]
        #     # print("state {}".format(state.shape))
        # dist = self.actor.forward(state, std=self.std)
        # if eval_mode:
        #     action = dist.mean
        # else:
        #     action = dist.sample(clip=None)
        # return action


# def update_q(
#     optim: torch.optim.Optimizer,
#     actor: Actor,
#     critic: Critic,
#     critic_target: Critic,
#     std_clip: float = 0.3,
# ):
#     def update_q_fn(
#         state: State,
#         action: Action,
#         reward: float,
#         discount: float,
#         next_state: State,
#         std: float,
#     ):
#         with torch.no_grad():
#             next_action = actor(next_state, std=std).sample(clip=std_clip)

#             td_target = reward + discount * torch.min(
#                 *critic_target(next_state, next_action)
#             )

#         q1, q2 = critic(state, action)
#         q_loss = torch.mean(util.mse(q1, td_target) + util.mse(q2, td_target))

#         optim.zero_grad(set_to_none=True)
#         q_loss.backward()
#         optim.step()

#         return {"q": q1.mean().item(), "q_loss": q_loss.item()}

#     return update_q_fn


# def update_actor(
#     optim: torch.optim.Optimizer, actor: Actor, critic: Critic, std_clip: float = 0.3
# ):
#     def update_actor_fn(state: State, std: float):
#         action = actor(state, std=std).sample(clip=std_clip)
#         Q = torch.min(*critic(state, action))
#         actor_loss = -Q.mean()

#         optim.zero_grad(set_to_none=True)
#         actor_loss.backward()
#         optim.step()

#         return {"actor_loss": actor_loss.item()}

#     return update_actor_fn
