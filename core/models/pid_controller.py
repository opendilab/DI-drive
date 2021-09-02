import os
import sys
from collections import deque
import numpy as np
from easydict import EasyDict


class PIDController(object):
    """
    PID controller for speed and angle control in DI-drive.

    :Arguments:
        - K_P (float): P value of PID.
        - K_I (float): I value of PID.
        - K_D (float): D value of PID.
        - fps (int): Frame per second in simulation. Ised to carculate derivative.
        - n (int): Length of integral window.

    :Interfaces: forward, clear
    """

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, fps=10, n=30, **kwargs):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._dt = 1.0 / fps
        self._n = n
        self._window = deque(maxlen=self._n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = sum(self._window) * self._dt
            derivative = (self._window[-1] - self._window[-2]) / self._dt
        else:
            integral = 0.0
            derivative = 0.0

        control = 0.0
        control += self._K_P * error
        control += self._K_I * integral
        control += self._K_D * derivative

        return control

    def forward(self, error: float) -> float:
        """
        Run one step of controller, return control value.

        :Returns:
            float: Control value.
        """
        return self.step(error)

    def clear(self):
        """
        Clear integral window.
        """
        self._window.clear()


class CustomController():
    """
    Controller used by LBC.
    """

    def __init__(self, controller_args, k=0.5, n=2, wheelbase=2.89, dt=0.1):
        self._wheelbase = wheelbase
        self._k = k

        self._n = n
        self._t = 0

        self._dt = dt
        self._controller_args = controller_args

        self._e_buffer = deque(maxlen=10)

    def run_step(self, alpha, cmd):
        self._e_buffer.append(alpha)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        Kp = self._controller_args[str(cmd)]["Kp"]
        Ki = self._controller_args[str(cmd)]["Ki"]
        Kd = self._controller_args[str(cmd)]["Kd"]

        return (Kp * alpha) + (Kd * _de) + (Ki * _ie)
