#!/usr/bin/env python3
import abc
import logging
from typing import Tuple, List
from copy import deepcopy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import src.agents.utils as util
import torch
import torch.nn as nn
from src.custom_types import (
    Action,
    Data,
    LatentState,
    Observation,
    Reward,
    State,
    StatePrediction,
)
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    @abc.abstractmethod
    def __call__(self, obs: Observation) -> LatentState:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def latent_dim(self) -> int:
        raise NotImplementedError


class TransitionModel(nn.Module):
    @abc.abstractmethod
    def __call__(self, latent_state: LatentState, action: Action) -> LatentState:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def latent_dim(self) -> int:
        raise NotImplementedError


class RewardModel(nn.Module):
    @abc.abstractmethod
    def __call__(self, latent_state: LatentState, action: Action) -> Reward:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def latent_dim(self) -> int:
        raise NotImplementedError


class MLPEncoder(Encoder):
    def __init__(self, obs_shape: Tuple[int], mlp_dims: List[int], latent_dim: int):
        super().__init__()
        self._encoder = util.mlp(obs_shape[0], mlp_dims, latent_dim)
        self.apply(util.orthogonal_init)
        self._latent_dim = latent_dim

    def __call__(self, obs: Observation) -> LatentState:
        latent_state = self._encoder(obs)
        return latent_state

    @property
    def latent_dim(self) -> int:
        return self._latent_dim


class MLPTransitionModel(TransitionModel):
    def __init__(self, latent_dim: int, mlp_dims: List[int], action_dim: int):
        super().__init__()
        self._network = util.mlp(latent_dim + action_dim, mlp_dims, latent_dim)
        self.apply(util.orthogonal_init)
        self._latent_dim = latent_dim

    def __call__(self, latent_state: LatentState, action: Action) -> LatentState:
        state_action_input = torch.concat([latent_state, action], -1)
        return self._network(state_action_input)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim


class MLPRewardModel(RewardModel):
    def __init__(self, latent_dim: int, mlp_dims: List[int], action_dim: int):
        super().__init__()
        self._network = util.mlp(latent_dim + action_dim, mlp_dims, 1)
        self.apply(util.orthogonal_init)
        self._latent_dim = latent_dim

    def __call__(self, latent_state: LatentState, action: Action) -> Reward:
        state_action_input = torch.concat([latent_state, action], -1)
        return self._network(state_action_input)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        device: str = "cuda",
    ):
        if "cuda" in device:
            encoder.cuda()
            transition_model.cuda()
            reward_model.cuda()

        assert encoder.latent_dim == transition_model.latent_dim
        assert encoder.latent_dim == reward_model.latent_dim

        self.encoder = encoder
        self.encoder_targ = deepcopy(encoder)  # TODO does this copy correctly?
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.device = device

    @torch.no_grad()
    def __call__(self, obs: Observation, action: Action) -> Tuple[LatentState, Reward]:
        latent_state = self.encoder(obs)
        next_latent_state = self.transition_model.forward(
            latent_state=latent_state, action=action
        )
        next_reward = self.reward_model.forward(
            latent_state=latent_state, action=action
        )
        return next_latent_state, next_reward
