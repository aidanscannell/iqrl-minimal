#!/usr/bin/env python3
import abc
from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from gymnasium.spaces import Box, Space
from jaxtyping import Float
from stable_baselines3.common.buffers import ReplayBuffer
from torch import Tensor


# from torchtyping import TensorType

Observation = Float[Tensor, "obs_dim"]
# State = Float[Tensor, "state_dim"]
Latent = Float[Tensor, "latent_dim"]
Action = Float[Tensor, "action_dim"]
#
BatchObservation = Float[Observation, "batch"]
# BatchState = Float[State, "batch"]
BatchLatent = Float[Latent, "batch"]
BatchAction = Float[Action, "batch"]

Value = Float[Tensor, ""]
BatchValue = Float[Value, "batch_size"]

# InputData = TensorType["num_data", "input_dim"]
# OutputData = TensorType["num_data", "output_dim"]
# Data = Tuple[InputData, OutputData]

# State = TensorType["batch_size", "state_dim"]
# Action = TensorType["batch_size", "action_dim"]

# ActionTrajectory = TensorType["horizon", "batch_size", "action_dim"]
# StateTrajectory = TensorType["horizon", "batch_size", "state_dim"]

EvalMode = bool
T0 = bool

# StateMean = TensorType["batch_size", "state_dim"]
# StateVar = TensorType["batch_size", "state_dim"]

# DeltaStateMean = TensorType["batch_size", "state_dim"]
# DeltaStateVar = TensorType["batch_size", "state_dim"]
# NoiseVar = TensorType["state_dim"]

# RewardMean = TensorType["batch_size"]
# RewardVar = TensorType["batch_size"]

# Input = TensorType["batch_size, input_dim"]
# OutputMean = TensorType["batch_size, output_dim"]
# OutputVar = TensorType["batch_size, output_dim"]


# @dataclass
class Agent(abc.ABC, nn.Module):
    # observation_space: Space
    # action_space: Box
    # name: str = "BaseAgent"

    def __init__(
        self, observation_space: Space, action_space: Box, name: str = "BaseAgent"
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.name = name

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

    @torch.compile
    def predict(
        self, observation, state=None, episode_start=None, deterministic: bool = False
    ):
        """Allows agent to work with Stablebaselines evaluate_policy"""
        action = self.select_action(observation=observation, eval_mode=True)
        # action = self.select_action(observation=observation, eval_mode=True, t0=t0)
        recurrent_state = None
        return action, recurrent_state
