import numpy as np
from numpy import linalg
from scipy.optimize import minimize
from typing import Dict, List


class FollowState:

    def __init__(self, x=0, y=0, th=0, v=0, cte=0, eth=0) -> None:
        self.x = x
        self.y = y
        self.th = th
        self.v = v
        self.cte = cte
        self.eth = eth


class ControlInput:

    def __init__(self, steer_angle: float = 0, acceleration: float = 0) -> None:
        self.steer_angle = steer_angle
        self.acceleration = acceleration


def wrap_angle(angle_in_degree):
    angle_in_rad = angle_in_degree / 180.0 * np.pi
    while (angle_in_rad > np.pi):
        angle_in_rad -= 2 * np.pi
    while (angle_in_rad <= -np.pi):
        angle_in_rad += 2 * np.pi
    return angle_in_rad


class MPCController(object):
    """
    Model Predictive Control (MPC) method for speed and angle control in DI-drive. MPC takes a target
    waypoints list as input. It will minimize the objective function that calculates the loss between
    target trajectory & speed and predicted vehicle states in future.

    :Arguments:
        - args_objective (Dict, optional): Args in objective function containing loss weights. Defaults to None.
        - horizon (int, optional): Steps MPC model predicts in calculating objective function. Defaults to 10.
        - fps (int, optional): FPS of predictive model. Defaults to 5.
    """

    def __init__(self, args_objective: Dict = None, horizon: int = 10, fps: int = 5) -> None:
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._waypoints_x = None
        self._waypoints_y = None
        self._target_speed = 0

        if args_objective is None:
            args_objective = {
                'CTE_W': 1,
                'ETH_W': 2,
                'V_W': 1,
                'ST_W': 0.5,
                'ACC_W': 0.01,
            }
        self._cte_w = args_objective['CTE_W']
        self._eth_w = args_objective['ETH_W']
        self._v_w = args_objective['V_W']
        self._st_w = args_objective['ST_W']
        self._acc_w = args_objective['ACC_W']

        self._horizon = horizon
        self._dt = 1.0 / fps

    def _model(self, control_input: ControlInput, input_state: FollowState, coeff: List) -> FollowState:
        L = 2.871
        steer = control_input.steer_angle
        acc = control_input.acceleration
        output_state = FollowState()
        output_state.x = input_state.x + input_state.v * np.cos(input_state.th) * self._dt
        output_state.y = input_state.y + input_state.v * np.sin(input_state.th) * self._dt
        output_state.th = input_state.th + (input_state.v / L) * steer * self._dt
        output_state.v = input_state.v + acc * self._dt

        th_des = np.arctan(coeff[2] + 2 * coeff[1] * input_state.x + 3 * coeff[0] * input_state.x ** 2)
        output_state.cte = np.polyval(coeff, input_state.x
                                      ) - input_state.y + (input_state.v * np.sin(input_state.eth) * self._dt)
        output_state.eth = input_state.th - th_des + ((input_state.v / L) * steer * self._dt)
        return output_state

    def _objective(self, u, *args):
        state = args[0]
        coeff = args[1]
        cost = 0.0
        control_input = ControlInput()

        for i in range(self._horizon):
            control_input.acceleration = u[i * 2]
            control_input.steer_angle = u[i * 2 + 1]
            state = self._model(control_input, state, coeff)
            cost += self._cte_w * state.cte ** 2
            cost += self._eth_w * state.eth ** 2
            cost += self._v_w * (state.v - self._target_speed) ** 2
            cost += self._st_w * u[i * 2 + 1] ** 2
            cost += self._acc_w * u[i * 2] ** 2
        return cost

    def _map_waypoints_to_car_coord(self, waypoints):
        wps = np.squeeze(waypoints)
        wps_x = wps[:, 0]
        wps_y = wps[:, 1]

        cos_yaw = np.cos(-self._current_yaw)
        sin_yaw = np.sin(-self._current_yaw)

        self._waypoints_x = cos_yaw * (wps_x - self._current_x) - sin_yaw * (wps_y - self._current_y)
        self._waypoints_y = sin_yaw * (wps_x - self._current_x) + cos_yaw * (wps_y - self._current_y)
        np.append(self._waypoints_x, [0, 0.01])
        np.append(self._waypoints_y, [0, 0])

    def step(self):
        steer = 0
        throttle = 0
        brake = 0
        bounds = []
        for i in range(self._horizon):
            bounds += [[-5, 5]]
            bounds += [[-0.8, 0.8]]
        init_state = FollowState(0, 0, 0, self._current_speed)
        u = np.zeros(self._horizon * 2)
        waypoint_coeff = np.polyfit(self._waypoints_x, self._waypoints_y, 3)

        mpc_solution = minimize(
            self._objective,
            u,
            (init_state, waypoint_coeff),
            method='SLSQP',
            bounds=bounds,
            tol=0.1,
        )
        actual_control = mpc_solution.x
        steer_output = actual_control[1] * 180 / 70 / np.pi
        acc_output = actual_control[0] / 5 + 0.3

        if acc_output < 0:
            brake = abs(acc_output)
        else:
            throttle = acc_output
        steer = steer_output
        return steer, throttle, brake

    def forward(self, ego_pose: List, target_speed: float, waypoints: List) -> Dict:
        """
        Run one step of controller, return the control signal.

        :Arguments:
            - ego_pose (List): Current location of ego vehicle: [x, y, yaw, speed].
            - target_speed (float): Target driving speed.
            - waypoints (List): Target trajectory waypoints.

        :Returns:
            Dict: Control signal containing steer, throttle and brake.
        """
        self._current_x = ego_pose[0]
        self._current_y = ego_pose[1]
        self._current_yaw = wrap_angle(ego_pose[2])
        self._current_speed = ego_pose[3]

        self._target_speed = target_speed
        self._map_waypoints_to_car_coord(waypoints)

        steer, throttle, brake = self.step()
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        control = dict()
        control['steer'] = steer
        control['throttle'] = throttle
        control['brake'] = brake
        return control
