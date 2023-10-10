#!/usr/bin/env python3
import abc
from typing import Any, Optional

from gymnasium.spaces import Box, Space
from src.custom_types import Action, EvalMode, Observation, T0
from stable_baselines3.common.buffers import ReplayBuffer


# Data = Any


class Agent:
    def __init__(
        self, observation_space: Space, action_space: Box, name: str = "BaseAgent"
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.name = name

    @abc.abstractmethod
    def select_action(
        self, observation, eval_mode: EvalMode = False, t0: T0 = None
    ) -> Action:
        raise NotImplementedError

    @abc.abstractmethod
    def train(
        self, replay_buffer: ReplayBuffer, num_updates: Optional[int] = None
    ) -> Optional[dict]:
        raise NotImplementedError

    # def update(self, data_new: Data):
    #     pass

    def predict(
        self, observation, state=None, episode_start=None, deterministic: bool = False
    ):
        """Allows agent to work with Stablebaselines evaluate_policy"""
        action = self.select_action(observation=observation, eval_mode=True)
        # action = self.select_action(observation=observation, eval_mode=True, t0=t0)
        recurrent_state = None
        return action, recurrent_state
