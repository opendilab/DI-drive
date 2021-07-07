# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
from typing import Dict, List
import numpy as np
import carla
from core.simulators.carla_data_provider import CarlaDataProvider


class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side.

    The arguments dictionary of PID controller use the following semantics:

        - K_P -- Proportional term
        - K_D -- Differential term
        - K_I -- Integral term
        - dt -- Time step in seconds

    :Arguments:
        - args_lateral (Dict): dictionary of arguments to set the lateral PID controller.
        - args_longitudinal (Dict): dictionary of arguments to set the longitudinal PID controller.
        - max_throttle (float, optional): Max value of throttle. Defaults to 0.75.
        - max_brake (float, optional): Max value of brake. Defaults to 0.3.
        - max_steering (float, optional): Max steering. Defaults to 0.8.

    :Interfaces: forward
    """

    def __init__(
        self,
        args_lateral: Dict,
        args_longitudinal: Dict,
        max_throttle: float = 0.75,
        max_brake: float = 0.3,
        max_steering: float = 0.8
    ):
        """
        Constructor method.
        """
        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self.past_steering = 0
        self._lon_controller = PIDLongitudinalController(**args_longitudinal)
        self._lat_controller = PIDLateralController(**args_lateral)

    def forward(
            self, current_speed: float, current_loc: List, current_ori: List, target_speed: float, target_loc: List
    ) -> Dict:
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed. Thw steering change between to step must be within 0.1.

        :Arguments:
            - current_speed (float): Current hero vehicle speed in km/h.
            - current_loc (List): Current hero vehicle coordinate location.
            - current_ori (List): Current hero vehicle orientation vector.
            - target_speed (float): Target speed in km/h
            - target_loc (List): Target coordinate location.

        :Returns:
            Dict: Control signal containing steer, throttle and brake.
        """

        acceleration = self._lon_controller.run_step(current_speed, target_speed)
        current_steering = self._lat_controller.run_step(current_loc, current_ori, target_loc)
        control = dict()
        if acceleration >= 0.0:
            control['throttle'] = min(acceleration, self.max_throt)
            control['brake'] = 0.0
        else:
            control['throttle'] = 0.0
            control['brake'] = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control['steer'] = steering
        self.past_steering = steering

        return control


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, current_speed, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.
        """

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)


class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, current_location, current_vector, target_location):
        v_vec = np.array([current_vector[0], current_vector[1], 0.0])
        w_vec = np.array([target_location[0] - current_location[0], target_location[1] - current_location[1], 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)
