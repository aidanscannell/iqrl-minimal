#!/usr/bin/env python3
import logging
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import helper as h
import numpy as np
import torch
import torch.nn as nn
from custom_types import BatchAction, BatchObservation, BatchValue
from gymnasium.spaces import Box, Space


class Critic(nn.Module):
    def forward(self, observation: BatchObservation, action: BatchAction) -> BatchValue:
        raise NotImplementedError


class MLPCritic(Critic):
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
        self._critic = h.mlp(
            in_dim=input_dim, mlp_dims=mlp_dims, out_dim=1, act_fn=act_fn
        )
        self.reset(full_reset=True)

    def forward(self, observation: BatchObservation, action: BatchAction) -> BatchValue:
        x = torch.cat([observation, action], -1)
        q_value = self._critic(x)
        return q_value

    def reset(self, full_reset: bool = False):
        if full_reset:
            self.apply(h.orthogonal_init)
        else:
            params = list(self.parameters())
            h.orthogonal_init(params[-2:])
