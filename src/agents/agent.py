#!/usr/bin/env python3
import abc
from typing import Any, Optional

from src.custom_types import Action, EvalMode, Observation, T0
from src.utils.buffer import ReplayBuffer


Data = Any


class Agent:
    @abc.abstractmethod
    def select_action(self, obs: Observation, eval_mode: EvalMode, t0: T0) -> Action:
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, replay_buffer: ReplayBuffer) -> Optional[dict]:
        raise NotImplementedError

    def update(self, data_new: Data):
        pass
