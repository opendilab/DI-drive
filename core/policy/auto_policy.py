import numpy as np
from typing import List, Dict, Optional

from .base_carla_policy import BaseCarlaPolicy
from core.models import VehiclePIDController, SteerNoiseWrapper, MPCController
from core.utils.others.config_helper import deep_merge_dicts
from ding.torch_utils.data_helper import to_ndarray

DEFAULT_LATERAL_DICT = {'K_P': 1, 'K_D': 0.1, 'K_I': 0, 'dt': 0.1}
DEFAULT_LONGITUDINAL_DICT = {'K_P': 1.0, 'K_D': 0, 'K_I': 0.05, 'dt': 0.1}
DEFAULT_LAT_HW_DICT = {'K_P': 0.75, 'K_D': 0.01, 'K_I': 0.2, 'dt': 0.1}
DEFAULT_LAT_CITY_DICT = {'K_P': 0.58, 'K_D': 0.01, 'K_I': 0.25, 'dt': 0.1}
DEFAULT_LONG_HW_DICT = {'K_P': 0.37, 'K_D': 0.012, 'K_I': 0.016, 'dt': 0.1}
DEFAULT_LONG_CITY_DICT = {'K_P': 0.15, 'K_D': 0.025, 'K_I': 0.035, 'dt': 0.1}

DEFAULT_MPC_ARGS = {'CTE_W': 1, 'ETH_W': 2, 'V_W': 1, 'ST_W': 0.5, 'ACC_W': 0.01}


class AutoPIDPolicy(BaseCarlaPolicy):
    """
    Autonomous Driving policy follows target waypoint in env observations. It uses a Vehicle PID controller
    for each env with a specific env id related to it. In each updating, all envs should use the correct env id
    to make the PID controller works well, and the controller should be reset when starting a new episode.

    The policy has 2 modes: `collect` and `eval`. Their interfaces operate in the same way. The only difference is
    that in `collect` mode the ``forward`` method may add noises to steer if set in config.

    :Arguments:
        - cfg (Dict): Config Dict.

    :Interfaces: reset, forward
    """

    config = dict(
        target_speed=25,
        max_brake=0.3,
        max_throttle=0.75,
        max_steer=0.8,
        ignore_light=False,
        ignore_speed_limit=False,
        tl_threshold=10,
        lateral_dict=DEFAULT_LATERAL_DICT,
        longitudinal_dict=DEFAULT_LONGITUDINAL_DICT,
        noise=False,
        noise_kwargs=dict(
            noise_len=5,
            drive_len=100,
            noise_type='uniform',
            noise_args=dict(
                low=-0.3,
                high=0.3,
            ),
        ),
        debug=False,
    )

    def __init__(
            self,
            cfg: Dict,
    ) -> None:
        super().__init__(cfg, enable_field=set(['collect', 'eval']))
        self._controller_dict = dict()
        self._last_steer_dict = dict()

        self.target_speed = self._cfg.target_speed
        self._max_brake = self._cfg.max_brake
        self._max_throttle = self._cfg.max_throttle
        self._max_steer = self._cfg.max_steer
        self._ignore_traffic_light = self._cfg.ignore_light
        self._ignore_speed_limit = self._cfg.ignore_speed_limit
        self._tl_threshold = self._cfg.tl_threshold

        self._lateral_dict = self._cfg.lateral_dict
        self._longitudinal_dict = self._cfg.longitudinal_dict

        self._debug = self._cfg.debug

    def _reset(self, data_id: int, noise: bool = False) -> None:
        if data_id in self._controller_dict:
            self._controller_dict.pop(data_id)

        if self._lateral_dict is None:
            if self.target_speed > 50:
                self._lateral_dict = DEFAULT_LAT_HW_DICT
            else:
                self._lateral_dict = DEFAULT_LAT_CITY_DICT
        if self._longitudinal_dict is None:
            if self.target_speed > 50:
                self._longitudinal_dict = DEFAULT_LONG_HW_DICT
            else:
                self._longitudinal_dict = DEFAULT_LONG_CITY_DICT

        controller = VehiclePIDController(
            args_lateral=self._lateral_dict,
            args_longitudinal=self._longitudinal_dict,
            max_throttle=self._max_throttle,
            max_brake=self._max_brake,
            max_steering=self._max_steer,
        )
        if noise:
            noise_controller = SteerNoiseWrapper(
                model=controller,
                **self._cfg.noise_kwargs,
            )
            self._controller_dict[data_id] = noise_controller
        else:
            self._controller_dict[data_id] = controller
        self._last_steer_dict[data_id] = 0

    def _forward(self, data_id: int, obs: Dict) -> Dict:
        controller = self._controller_dict[data_id]
        if obs['command'] == -1:
            control = self._emergency_stop(data_id)
        elif obs['agent_state'] == 2 or obs['agent_state'] == 3 or obs['agent_state'] == 5:
            control = self._emergency_stop(data_id)
        elif not self._ignore_traffic_light and obs['agent_state'] == 4:
            control = self._emergency_stop(data_id)
        elif not self._ignore_traffic_light and obs['tl_state'] in [0, 1] and obs['tl_dis'] < self._tl_threshold:
            control = self._emergency_stop(data_id)
        else:
            current_speed = obs['speed']
            current_location = obs['location']
            current_vector = obs['forward_vector']
            target_location = obs['target']
            if not self._ignore_speed_limit:
                target_speed = min(self.target_speed, obs['speed_limit'])
            else:
                target_speed = self.target_speed
            control = controller.forward(
                current_speed,
                current_location,
                current_vector,
                target_speed,
                target_location,
            )
            if abs(control['steer'] > 0.1 and current_speed > 15):
                control['throttle'] = min(control['throttle'], 0.3)
            self._last_steer_dict[data_id] = control['steer']
        return control

    def _emergency_stop(self, data_id: int) -> Dict:
        control = {
            'steer': self._last_steer_dict[data_id],
            'throttle': 0.0,
            'brake': 1.0,
        }
        return control

    def _forward_eval(self, data: Dict) -> Dict:
        """
        Running forward to get control signal of `eval` mode.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: Control dict stored in values for each provided env id.
        """
        data = to_ndarray(data)
        actions = dict()
        for i in data.keys():
            obs = data[i]
            action = self._forward(i, obs)
            actions[i] = {'action': action}
        return actions

    def _forward_collect(self, data: Dict) -> Dict:
        """
        Running forward to get control signal of `collect` mode.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: Control dict stored in values for each provided env id.
        """
        data = to_ndarray(data)
        actions = dict()
        for i in data.keys():
            obs = data[i]
            action = self._forward(i, obs)
            actions[i] = {'action': action}
        return actions

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        """
        Reset policy of `eval` mode. It will reset the controllers in providded env id.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        if data_id is not None:
            for id in data_id:
                self._reset(id)
        else:
            for id in self._controller_dict:
                self._reset(id)

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        """
        Reset policy of `collect` mode. It will reset the controllers in provided env id. Noise will be add
        to the controller according to config.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        noise = self._cfg.noise
        if data_id is not None:
            for id in data_id:
                self._reset(id, noise)
        else:
            for id in self._controller_dict:
                self._reset(id, noise)


class AutoMPCPolicy(BaseCarlaPolicy):
    """
    Autonomous Driving policy follows target waypoint list in env observations. It uses an MPC controller
    for each env with a specific env id related to it. In each updating, all envs should use the correct
    env id to make the MPC controller works well, and the controller should be reset when starting a new
    episode.

    The policy has 2 modes: `collect` and `eval`. Their interfaces operate in the same way. The only difference is
    that in `collect` mode the ``forward`` method may add noises to steer if set in config.

    :Arguments:
        - cfg (Dict): Config Dict.

    :Interfaces: reset, forward
    """

    config = dict(
        target_speed=25,
        mpc_args=None,
        ignore_light=False,
        ignore_speed_limit=False,
        horizon=5,
        fps=4,
        noise=False,
        debug=False,
    )

    def __init__(
            self,
            cfg: Dict,
    ) -> None:
        super().__init__(cfg, enable_field=set(['collect', 'eval']))
        self._controller_dict = dict()
        self._last_steer_dict = dict()

        self.target_speed = self._cfg.target_speed
        self._ignore_traffic_light = self._cfg.ignore_light
        self._ignore_speed_limit = self._cfg.ignore_speed_limit

        if self._cfg.mpc_args is None:
            self._mpc_args = DEFAULT_MPC_ARGS
        else:
            self._mpc_args = self._cfg.mpc_args

        self._horizon = self._cfg.horizon
        self._fps = self._cfg.fps

        self._debug = self._cfg.debug

    def _reset(self, data_id: int, noise: bool = False) -> None:
        if data_id in self._controller_dict:
            self._controller_dict.pop(data_id)

        controller = MPCController(self._mpc_args, self._horizon, self._fps)

        if noise:
            noise_controller = SteerNoiseWrapper(
                model=controller,
                noise_type='uniform',
                noise_kwargs={
                    'low': -0.3,
                    'high': 0.3
                },
            )
            self._controller_dict[data_id] = noise_controller
        else:
            self._controller_dict[data_id] = controller
        self._last_steer_dict[data_id] = 0

    def _forward(self, data_id: int, obs: Dict) -> Dict:
        controller = self._controller_dict[data_id]
        if obs['command'] == -1:
            control = self._emergency_stop(data_id)
        elif obs['agent_state'] == 2 or obs['agent_state'] == 3 or obs['agent_state'] == 5:
            control = self._emergency_stop(data_id)
        elif not self._ignore_traffic_light and obs['agent_state'] == 4:
            control = self._emergency_stop(data_id)
        elif not self._ignore_traffic_light and obs['tl_state'] in [0, 1] and obs['tl_dis'] < 10:
            control = self._emergency_stop(data_id)
        else:
            ego_pose = [
                obs['location'][0],
                obs['location'][1],
                obs['rotation'][1],
                obs['speed'] / 3.6,
            ]
            waypoints = obs['waypoint_list']
            if not self._ignore_speed_limit:
                target_speed = min(self.target_speed, obs['speed_limit'])
            else:
                target_speed = self.target_speed
            control = controller.forward(
                ego_pose,
                target_speed / 3.6,
                waypoints,
            )
            self._last_steer_dict[data_id] = control['steer']
        return control

    def _emergency_stop(self, data_id: int) -> Dict:
        control = {
            'steer': self._last_steer_dict[data_id],
            'throttle': 0.0,
            'brake': 1.0,
        }
        return control

    def _forward_eval(self, data: Dict) -> Dict:
        """
        Running forward to get control signal of `eval` mode.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: Control dict stored in values for each provided env id.
        """
        data = to_ndarray(data)
        actions = dict()
        for i in data.keys():
            obs = data[i]
            action = self._forward(i, obs)
            actions[i] = {'action': action}
        return actions

    def _forward_collect(self, data: Dict) -> Dict:
        """
        Running forward to get control signal of `collect` mode.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: Control dict stored in values for each provided env id.
        """
        data = to_ndarray(data)
        actions = dict()
        for i in data.keys():
            obs = data[i]
            action = self._forward(i, obs)
            actions[i] = {'action': action}
        return actions

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        """
        Reset policy of `eval` mode. It will reset the controllers in providded env id.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        if data_id is not None:
            for id in data_id:
                self._reset(id)
        else:
            for id in self._controller_dict:
                self._reset(id)

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        """
        Reset policy of `collect` mode. It will reset the controllers in provided env id. Noise will be add
        to the controller according to config.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        noise = self._cfg.noise
        if data_id is not None:
            for id in data_id:
                self._reset(id, noise)
        else:
            for id in self._controller_dict:
                self._reset(id, noise)
