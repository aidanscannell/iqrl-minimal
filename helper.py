#!/usr/bin/env python3
import re
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


def simnorm(z, V=8):
    shape = z.shape
    # print(f"shape {shape}")
    # print(f"first z {z.shape}")
    z = z.view(*shape[:-1], -1, V)
    # print(f"z after first view {z.shape}")
    z = torch.softmax(z, dim=-1)
    # print(f"z after softmax {z.shape}")
    z = z.view(*shape)
    # print(f"z after view {z.shape}")
    return z


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def __call__(self, val):
        if val < self.min_val:
            self.min_val = val
            self.counter = 0
        elif val > (self.min_val + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.min_val = np.inf


# __REDUCE__ = lambda b: "mean" if b else "none"


# def mse(pred, target, reduce=False):
#     """Computes the MSE loss between predictions and targets."""
#     return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def soft_update_params(model, model_target, tau: float):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for params, params_target in zip(model.parameters(), model_target.parameters()):
            params_target.data.lerp_(params.data, tau)
            # One below is from CleanRL
            # params_target.data.copy_(tau * params.data + (1 - tau) * params_target.data)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, Mish activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, \
        out_features={self.out_features}, \
        bias={self.bias is not None}{repr_dropout}, \
        act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act_fn=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]

    dims = [int(in_dim)] + mlp_dims + [int(out_dim)]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(
        NormedLinear(dims[-2], dims[-1], act=act_fn)
        if act_fn
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # elif isinstance(m, EnsembleLinear):
    #     for w in m.weight.data:
    #         nn.init.orthogonal_(w)
    #     if m.bias is not None:
    #         for b in m.bias.data:
    #             nn.init.zeros_(b)
    elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        # nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LinearSchedule:
    def __init__(self, start: float, end: float, num_steps: int):
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self.step_idx = 0
        self.values = np.linspace(start, end, num_steps)

    def __call__(self):
        return self.values[self.step_idx]

    def step(self):
        if self.step_idx < self.num_steps - 1:
            self.step_idx += 1


# def linear_schedule(schdl, step):
#     """
#     Outputs values following a linear decay schedule.
#     Adapted from https://github.com/facebookresearch/drqv2
#     """
#     try:
#         return float(schdl)
#     except ValueError:
#         match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
#         if match:
#             init, final, duration = [float(g) for g in match.groups()]
#             mix = np.clip(step / duration, 0.0, 1.0)
#             return (1.0 - mix) * init + mix * final
#     raise NotImplementedError(schdl)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
