#!/usr/bin/env python3
import abc
from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn
from gymnasium.spaces import Box, Space
from src.custom_types import (
    Action,
    EvalMode,
    Observation,
    T0,
    Value,
    BatchObservation,
    BatchAction,
    BatchValue,
)
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class Agent(abc.ABC):
    observation_space: Space
    action_space: Box
    name: str = "BaseAgent"

    @abc.abstractmethod
    def select_action(
        self, observation, eval_mode: EvalMode = False, t0: T0 = None
    ) -> Action:
        raise NotImplementedError

    @abc.abstractmethod
    def update(
        self, replay_buffer: ReplayBuffer, num_new_transitions: int
    ) -> Optional[dict]:
        raise NotImplementedError

    def predict(
        self, observation, state=None, episode_start=None, deterministic: bool = False
    ):
        """Allows agent to work with Stablebaselines evaluate_policy"""
        action = self.select_action(observation=observation, eval_mode=True)
        # action = self.select_action(observation=observation, eval_mode=True, t0=t0)
        recurrent_state = None
        return action, recurrent_state


class Critic(nn.Module):
    def forward(self, observation: BatchObservation, action: BatchAction) -> BatchValue:
        raise NotImplementedError


class Actor(nn.Module):
    def forward(self, observation: BatchObservation) -> BatchAction:
        raise NotImplementedError


# @dataclass
# class ActorCritic(Agent, abc.ABC):
class ActorCritic(Agent, abc.ABC):
    observation_space: Space
    action_space: Box
    actor: Actor
    critic: Critic
    name: str = "BaseActorCriticAgent"
