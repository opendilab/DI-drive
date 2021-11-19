import torch
import numpy as np
from typing import Dict, Any, List
import math
import gym

from core.envs import BaseCarlaEnv
from ding.torch_utils.data_helper import to_ndarray


DEFAULT_ACC_LIST = [
    (0, 1),
    (0.25, 0),
    (0.75, 0),
]
DEFAULT_STEER_LIST = [
    -0.8,
    -0.5,
    -0.2,
    0,
    0.2,
    0.5,
    0.8,
]


class DiscreteEnvWrapper(gym.Wrapper):

    def __init__(self, env: BaseCarlaEnv, acc_list: List = None, steer_list: List = None) -> None:
        super().__init__(env)
        self._acc_list = acc_list
        if acc_list is None:
            self._acc_list = DEFAULT_ACC_LIST
        self._steer_list = steer_list
        if steer_list is None:
            self._steer_list = DEFAULT_STEER_LIST
    
    def reset(self, *args, **kwargs) -> Any:
        obs = super().reset(*args, **kwargs)
        obs_out = {
            'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
            'speed': (obs['speed'] / 25).astype(np.float32),
        }
        return obs_out
    
    def step(self, id):
        if isinstance(id, torch.Tensor):
            id = id.item()
        id = np.squeeze(id)
        assert id < len(self._acc_list) * len(self._steer_list), (id, len(self._acc_list) * len(self._steer_list))
        mod_value = len(self._acc_list)
        acc = self._acc_list[id % mod_value]
        steer = self._steer_list[id // mod_value]
        action = {
            'steer': steer,
            'throttle': acc[0],
            'brake': acc[1],
        }
        obs, reward, done, info = super().step(action)
        obs_out = {
            'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
            'speed': (obs['speed'] / 25).astype(np.float32),
        }
        return obs_out, reward, done, info


class MultiDiscreteEnvWrapper(gym.Wrapper):

    def __init__(self, env: BaseCarlaEnv, acc_list: List = None, steer_list: List = None) -> None:
        super().__init__(env)
        self._acc_list = acc_list
        if acc_list is None:
            self._acc_list = DEFAULT_ACC_LIST
        self._steer_list = steer_list
        if steer_list is None:
            self._steer_list = DEFAULT_STEER_LIST

    def reset(self, *args, **kwargs) -> Any:
        obs = super().reset(*args, **kwargs)
        obs_out = {
            'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
            'speed': (obs['speed'] / 25).astype(np.float32),
        }
        return obs_out

    def step(self, action_ids):
        action_ids = to_ndarray(action_ids, dtype=int)
        action_ids = np.squeeze(action_ids)
        acc_id = action_ids[0]
        steer_id = action_ids[1]
        assert acc_id < len(self._acc_list), (acc_id, len(self._acc_list))
        assert steer_id < len(self._steer_list), (steer_id, len(self._steer_list))
        acc = self._acc_list[acc_id]
        steer = self._steer_list[steer_id]
        action = {
            'steer': steer,
            'throttle': acc[0],
            'brake': acc[1],
        }
        obs, reward, done, info = super().step(action)
        obs_out = {
            'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
            'speed': (obs['speed'] / 25).astype(np.float32),
        }
        return obs_out, reward, done, info


class ContinuousEnvWrapper(gym.Wrapper):

    def reset(self, *args, **kwargs) -> Any:
        obs = super().reset(*args, **kwargs)
        obs_out = {
            'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
            'speed': (obs['speed'] / 25).astype(np.float32),
        }
        return obs_out

    def step(self, action):
        action = to_ndarray(action)
        action = np.squeeze(action)
        steer = action[0]
        acc = action[1]
        if acc > 0:
            throttle, brake = acc, 0
        else:
            throttle, brake = 0, -acc

        action = {
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
        }
        obs, reward, done, info = super().step(action)
        obs_out = {
            'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
            'speed': (obs['speed'] / 25).astype(np.float32),
        }
        return obs_out, reward, done, info
