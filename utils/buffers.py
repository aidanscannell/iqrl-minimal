import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    next_state_discounts: th.Tensor


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self,
               batch_size: int,
               env: Optional[VecNormalize] = None,
               nstep: Optional[int] = 1,
               train_validation_split: Optional[float] = None,
               val: Optional[bool] = False,
               trains: Optional[np.ndarray] = None,
    ):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # We must ensure that the n-step samples are valid and from the same trajectory
        if train_validation_split is not None:
            if self.full and nstep > 2:
                idxs = (np.arange(0, self.buffer_size - nstep + 1) + self.pos) % self.buffer_size
            else:
                idxs = np.arange(0, self.buffer_size if self.full else (self.pos - nstep + 1))
            if not val:
                sample_idxs = idxs[trains[idxs]]
            else:
                sample_idxs = idxs[~trains[idxs]]
            batch_inds = np.random.choice(sample_idxs, size=batch_size, replace=False)
        elif self.full and nstep > 2:
            batch_inds = (np.random.randint(0, self.buffer_size - nstep + 1, size=batch_size) + self.pos) % self.buffer_size
        else:
            upper_bound = self.buffer_size if self.full else (self.pos - nstep + 1)
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    trains: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        nstep: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        discount: Optional[float] = 1.0,
        train_validation_split: Optional[float] = None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Set n-step returns
        self.nstep = nstep
        self.discount = discount
        self.train_validation_split = train_validation_split

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.trains = np.zeros((self.buffer_size), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def resample_val(self) -> None:
        assert self.train_validation_split is not None
        self.trains = np.random.rand(len(self.trains)) < self.train_validation_split

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        timeout: np.ndarray,
        infos: List[Dict[str, Any]],
        trains: Optional[np.ndarray] = None,
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        
        if trains is not None and self.train_validation_split is not None:
            self.trains[self.pos] = np.array(trains)
        elif self.train_validation_split is not None:
            # If multiple environment, all get same trains value
            self.trains[self.pos] = np.random.rand() < self.train_validation_split 

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = timeout
#            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, val: Optional[bool] = False) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env, nstep=self.nstep, train_validation_split=self.train_validation_split, val=val, trains=self.trains)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.train_validation_split is not None:
            if self.full:
                idxs = (np.arange(1, self.buffer_size - self.nstep + 1) + self.pos) % self.buffer_size
            else:
                idxs = np.arange(0, self.pos - self.nstep + 1)
            if not val:
                sample_idxs = idxs[self.trains[idxs]]
            else:
                sample_idxs = idxs[~self.trains[idxs]]
            batch_inds = np.random.choice(sample_idxs, size=batch_size, replace=False)
        elif self.full:
            batch_inds = (np.random.randint(1, self.buffer_size - self.nstep + 1, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos - self.nstep + 1, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        acts = self.actions[batch_inds, env_indices, :]
        dones = np.zeros((*acts.shape[:-1], 1))
        next_obs = np.zeros_like(obs)
        timeout_or_dones = np.zeros_like(dones)
        rewards = np.zeros_like(dones)
        next_state_discounts = np.ones_like(dones)

        for n in range(self.nstep):
            idxs = (batch_inds + n) % self.buffer_size

            if self.optimize_memory_usage:
                next_obs = np.where(timeout_or_dones, next_obs, self._normalize_obs(self.observations[(idxs + 1) % self.buffer_size, env_indices, :], env))
            else:
                next_obs = np.where(timeout_or_dones, next_obs, self._normalize_obs(self.next_observations[idxs, env_indices, :], env))

            next_state_discounts *= np.where(timeout_or_dones, 1, self.discount)
            dones = np.where(timeout_or_dones, dones, self.dones[idxs, env_indices].reshape(-1, 1))
            rewards += np.where(timeout_or_dones, 0, self.discount**n * self._normalize_reward(self.rewards[idxs, env_indices].reshape(-1, 1), env))
            timeout_or_dones = np.logical_or(timeout_or_dones, np.logical_or(dones, self.timeouts[idxs, env_indices].reshape(-1, 1)))

        data = (obs, acts, next_obs, dones.astype(np.float32), rewards.astype(np.float32), next_state_discounts.astype(np.float32))

#        obs_new = obs
#        acts_new = acts
#        dones_new = dones
#        next_obs_new = next_obs
#        rewards_new = rewards
#        if self.optimize_memory_usage:
#            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
#        else:
#            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
#        data_old = (
#            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
#            self.actions[batch_inds, env_indices, :],
#            next_obs,
#            # Only use dones that are not due to timeouts
#            # deactivated by default (timeouts is initialized as an array of False)
#            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
#            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
#        )
#        obs = data_old[0]
#        acts = data_old[1]
#        next_obs = data_old[2]
#        dones = data_old[3]
#        rewards = data_old[4]
#        print(np.allclose(obs, obs_new))
#        print(np.allclose(acts, acts_new))
#        print(np.allclose(next_obs, next_obs_new))
#        print(np.allclose(dones, dones_new))
#        print(np.allclose(rewards, rewards_new))
#        exit(0)

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype
