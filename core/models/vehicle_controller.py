# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
from typing import Dict, List
import numpy as np


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
            - current_vec (List): Current hero vehicle forward vector.
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


class RWPFLateralController():

    def __init__(self, L=2.405, k_k=1.235, k_theta=0.456, k_e=0.11, alpha=1.8):
        self.L = L
        self.k_k = k_k
        self.k_theta = k_theta
        self.k_e = k_e
        self.alpha = alpha

    def rad_lim(self, rad):
        while (rad > np.pi):
            rad -= (2 * np.pi)
        while (rad < -np.pi):
            rad += (2 * np.pi)
        return rad

    def run_step(self, current_state, target_state):
        dx = current_state['x'] - target_state['x']
        dy = current_state['y'] - target_state['y']
        tx = np.cos(target_state['theta'])
        ty = np.sin(target_state['theta'])

        e = dx * ty - dy * tx
        #theta_e = self.rad_lim(current_state['theta'] - target_state['theta'])
        theta_e = self.rad_lim(-target_state['theta'])

        w1 = self.k_k * target_state['v'] * target_state['k'] * np.cos(theta_e)
        w2 = -self.k_theta * np.abs(current_state['v']) * theta_e
        w3 = (self.k_e * target_state['v'] * np.exp(-theta_e ** 2 / self.alpha)) * e
        w = (w1 + w2 + w3) * 0.8
        if current_state['v'] < 0.02:
            steer = 0
        else:
            steer = np.arctan2(w * self.L, current_state['v']) * 2 / np.pi

        #print(e, w, steer)
        return steer


class VehicleCapacController():

    def __init__(
        self,
        args_lateral: Dict,
        args_longitudinal: Dict,
        max_throttle: float = 0.8,
        max_brake: float = 0.5,
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
        self._lat_controller = RWPFLateralController(**args_lateral)

    def reset(self):
        self.past_steering = 0

    def forward(self, current_state: Dict, target_state: Dict) -> Dict:
        acceleration = self._lon_controller.run_step(current_state['v'], target_state['v'])
        current_steering = self._lat_controller.run_step(current_state, target_state)
        control = dict()
        control['throttle'] = np.clip(acceleration, 0, self.max_throt)
        control['brake'] = -np.clip(acceleration, -self.max_brake, 0)
        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        steering = np.clip(current_steering, -self.max_steer, self.max_steer)
        #print(steering, self.past_steering)

        control['steer'] = steering
        if control['throttle'] > 0 and abs(current_state['v']) < 0.1 and abs(target_state['v']) < 0.1:
            control['throttle'] = 0.
            control['brake'] = 1.
            control['steer'] = 0.

        self.past_steering = steering

        return control
