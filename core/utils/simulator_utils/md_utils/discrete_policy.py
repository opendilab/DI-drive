import logging

import numpy as np

from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math_utils import not_zero, wrap_to_pi, point_distance
from metadrive.utils.scene_utils import is_same_lane_index, is_following_lane_index

from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController
from metadrive.utils import clip
from metadrive.examples import expert
from metadrive.policy.env_input_policy import EnvInputPolicy
from direct.controls.InputState import InputState
from metadrive.engine.engine_utils import get_global_config

import gym
from gym import Wrapper
from gym import spaces
# from metadrive.envs.base_env import BaseEnv
# from metadrive.envs.metadrive_env import MetaDriveEnv
from typing import Callable


class ActionType(object):
    #def __init__(self, env: 'MetaDriveEnv', **kwargs) -> None:
    def __init__(self, env, **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    def vehicle_class(self) -> Callable:
        raise NotImplementedError

    def act(self, action) -> None:
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class DiscreteMetaAction(ActionType):
    ACTIONS_ALL = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER', 5: 'Holdon'}

    def __init__(self, **kwargs):
        self.actions = self.ACTIONS_ALL
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(5)
        #return spaces.Discrete(len(self.actions))

    # @property
    # def vehicle_class(self) -> Callable:
    #     return MDPVehicle

    def act(self, action: int) -> None:
        self.controlled_vehicle.act(self.actions[action])
