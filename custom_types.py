#!/usr/bin/env python3
from typing import NamedTuple, Tuple

from jaxtyping import Float
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
