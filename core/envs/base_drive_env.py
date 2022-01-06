'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''

from abc import ABC, abstractmethod
from typing import Any, Dict
import copy
from easydict import EasyDict
import gym
from gym import utils

from ding.utils.default_helper import deep_merge_dicts


class BaseDriveEnv(gym.Env, utils.EzPickle):
    """
    Base class for environments. It is inherited from `gym.Env` and uses the same interfaces.
    All Drive Env class is supposed to inherit from this class.

    Note:
        To run Reinforcement Learning on DI-engine platform, the environment should be wrapped with `DingEnvWrapper`.

    :Arguments:
        - cfg (Dict): Config Dict.

    :Interfaces: reset, step, close, seed
    """
    config = dict()

    @abstractmethod
    def __init__(self, cfg: Dict, **kwargs) -> None:
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        utils.EzPickle.__init__(self)

    @abstractmethod
    def step(self, action: Any) -> Any:
        """
        Run one step of the environment and return the observation dict.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs) -> Any:
        """
        Reset current environment.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Release all resources in environment and close.
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        """
        Set random seed.
        """
        raise NotImplementedError

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
