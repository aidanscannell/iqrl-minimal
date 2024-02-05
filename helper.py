#!/usr/bin/env python3
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.linalg import cond, matrix_rank
from vector_quantize_pytorch import FSQ


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

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py
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

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py
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


class FSQMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mlp_dims: List[int],
        levels: List[int] = [8, 8],  # target size 2^6, actual size 64
        out_dim: int = 1024,  # out_dim % levels == 0
    ):
        super().__init__()
        self.levels = levels
        self.out_dim = out_dim
        assert out_dim % len(levels) == 0
        self.latent_dim = int(out_dim / len(levels))
        self.mlp = mlp(
            in_dim=in_dim, mlp_dims=mlp_dims, out_dim=self.out_dim, act_fn=None
        )
        self._fsq = FSQ(levels)

    def forward(self, x, quantized: bool = False, both: bool = False):
        flag = False
        if x.ndim == 1:
            flag = True
            x = x.view(1, -1)
        leading_shape = x.shape[:-1]
        z = self.mlp(x)
        z = z.view(*leading_shape, self.latent_dim, len(self.levels))
        z, indices = self.fsq(z)
        z = z.view(*leading_shape, self.out_dim)

        if flag:
            z = z[0, ...]
            indices = indices[0, ...]
        if both:
            return z, indices
        elif quantized:
            return indices
        else:
            return z

    def fsq(self, z):
        if z.ndim > 3:
            z, indices = torch.func.vmap(self._fsq)(z)
        else:
            z, indices = self._fsq(z)
        return z, indices


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


def calc_rank(name, z):
    """Log rank of latent"""
    rank3 = matrix_rank(z, atol=1e-3, rtol=1e-3)
    rank2 = matrix_rank(z, atol=1e-2, rtol=1e-2)
    rank1 = matrix_rank(z, atol=1e-1, rtol=1e-1)
    condition = cond(z)
    info = {}
    for j, rank in enumerate([rank1, rank2, rank3]):
        info.update({f"{name}-rank-{j}": rank.item()})
    info.update({f"{name}-cond-num": condition.item()})
    return info


class MLPResettable(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(x)

    def reset(self, reset_type: str = "last-layer"):
        if reset_type in "full":
            orthogonal_init(self.parameters())
        elif reset_type in "last-layer":
            params = list(self.parameters())
            orthogonal_init(params[-2:])
        else:
            raise NotImplementedError
