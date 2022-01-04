'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description: Base simulator.
'''
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict
from easydict import EasyDict
import copy

from ding.utils.default_helper import deep_merge_dicts


class BaseSimulator(ABC):
    """
    Base class for simulators.

    :Arguments:
        - cfg (Dict): Config Dict

    :Interfaces: apply_control, run_step
    """

    config = dict()

    def __init__(self, cfg: Dict):
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg

    @abstractmethod
    def apply_control(self, control: Dict):
        """
        Apply control signal to hero vehicle. It will take effect in the next world tick.

        :Arguments:
            control (Dict): Control signal dict
        """
        raise NotImplementedError

    @abstractmethod
    def run_step(self):
        """
        Run one step for simulator.
        """
        raise NotImplementedError

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)
