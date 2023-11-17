"""An async vector environment."""
import os
import sys
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import dgl
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# from numpy.typing import NDArray

from psp.env.genv import GEnv as Env

# import gymnasium as gym
# from gymnasium.core import Env, ObsType
# from gymnasium.error import (
#     AlreadyPendingCallError,
#     ClosedEnvironmentError,
#     CustomSpaceError,
#     NoAsyncCallError,
# )
# from gymnasium.vector.utils import (
#     CloudpickleWrapper,
#     clear_mpi_env_vars,
#     concatenate,
#     create_empty_array,
#     create_shared_memory,
#     iterate,
#     read_from_shared_memory,
#     write_to_shared_memory,
# )
from .vector_env import GraphVectorEnv

__all__ = ["AsyncVectorEnv"]


class AsyncGraphVectorEnv(GraphVectorEnv):
    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        shared_memory: bool = False,
        copy: bool = True,
    ):
        self.nenv = len(env_fns)
        self.executor = ThreadPoolExecutor()
        self.env_fns = env_fns
        super().__init__(
            num_envs=len(env_fns),
        )

        self.copy = copy
        self.envs = []
        for i in range(self.nenv):
            self.envs.append(self.env_fns[i]())
        self.futures = {}

        self.observations = []

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for i in range(self.nenv):
            self.futures[i] = self.executor.submit(self.envs[i].reset)

    def reset_wait(
        self,
        timeout: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        observations_list, infos = [], {}
        successes = []

        for i in range(self.nenv):
            obs, info = self.futures[i].result()
            observations_list.append(obs)
            infos = self._add_info(infos, info, i)
        self.observations = observations_list

        self.futures.clear()

        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions):
        for i in range(self.nenv):
            self.futures[i] = self.executor.submit(self.envs[i].step, actions[i])

    def step_wait(self, timeout: Optional[Union[int, float]] = None):
        observations_list, rewards, terminateds, truncateds, infos = [], [], [], [], {}
        successes = []

        for i in range(self.nenv):
            (
                observation,
                rew,
                terminated,
                truncated,
                info,
            ) = self.futures[i].result()
            if terminated or truncated:
                old_observation, old_info = observation, info
                observation, info = self.envs[i].reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations_list.append(observation)
            rewards.append(rew)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos = self._add_info(infos, info, i)

        self.observations = observations_list
        self.futures.clear()

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            infos,
        )

    def call_async(self, name: str, *args, **kwargs):
        pass

    def call_wait(self, timeout: Optional[Union[int, float]] = None) -> list:
        pass

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        pass

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value = self.error_queue.get()
            print(
                f"Received the following error from Worker-{index}: {exctype.__name__}: {value}"
            )
            print(f"Shutting down Worker-{index}.")
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                print("Raising the last exception back to the main process.")
                raise exctype(value)

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        pass
