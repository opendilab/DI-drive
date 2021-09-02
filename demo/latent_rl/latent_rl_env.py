from ding.torch_utils.data_helper import to_tensor
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from core.envs import SimpleCarlaEnv
from core.utils.model_utils import common


class CarlaLatentRLEnv(SimpleCarlaEnv):
    new_config = dict(
        discrete_action=True,
        discrete_dim=10,
        stuck_thres=10,
    )
    config = SimpleCarlaEnv.config
    config.update(new_config)

    def __init__(self, cfg: Dict, host: str, port: int, tm_port: Optional[int] = None, **kwargs) -> None:
        super().__init__(cfg, host=host, port=port, tm_port=tm_port, **kwargs)
        self._dis_action = self._cfg.discrete_action
        self._dis_action_dim = self._cfg.discrete_dim
        self._stuck_thres = self._cfg.stuck_thres

        self._low_speed_count = 0
        self._previous_node = np.float32([-1, -1, -1])
        self._stuck_count = 0
        self._dis_reward = 0
        self._dead_reward = 0

    def get_observations(self) -> Dict:
        obs = super().get_observations()
        new_obs = dict()
        birdview = common.crop_birdview(obs['birdview'], dx=-10)
        new_obs['birdview'] = birdview[..., :7]
        ego_info = []
        cmd = np.eye(4)[obs['command'] - 1]
        ego_info.append(obs['forward_vector'])
        ego_info.append(obs['velocity'] / (30 / 3.6))  # 3, max 30 km / h
        ego_info.append(obs['acceleration'] / 10)  # 3
        ego_info = np.concatenate(ego_info)
        ego_info = np.concatenate([cmd, ego_info])
        new_obs['ego_info'] = ego_info

        return new_obs

    def _get_route(self, max_index=255):
        while True:
            start = np.random.randint(max_index)
            end = np.random.randint(max_index)
            if abs(start - end) > 50:
                return start, end

    def reset(self, col_is_failure=True, **kwargs) -> Dict:
        self._low_speed_count = 0
        self._stuck_count = 0
        start, end = self._get_route()
        env_params = {
            'weather': 1,
            'start': start,
            'end': end,
        }
        return super().reset(col_is_failure=col_is_failure, **env_params)

    def step(self, action: Dict) -> Tuple[Any, float, bool, Dict]:
        steer, throttle, brake = 0, 0, 0
        if not self._dis_action:
            # continuous action
            steer, throttle = action
        else:
            # discrete action
            idx = int(action)
            idx2act = lambda x: 2 / self._dis_action_dim * x - 1
            steer_idx = idx // self._dis_action_dim
            throttle_idx = idx % self._dis_action_dim
            steer = idx2act(steer_idx)
            throttle = idx2act(throttle_idx)
            throttle = (1 + throttle) / 2

        steer, throttle = float(steer), float(throttle)
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        action = {'steer': steer, 'throttle': throttle, 'brake': brake}
        return super().step(action)

    def is_failure(self):
        if self._low_speed_count > 50:
            return True
        if self._off_road:
            return True
        if self._dis_reward <= -1:
            return True
        if self._dead_reward <= -1:
            return True
        return False

    def compute_reward(self) -> Tuple[float, Dict]:
        goal_reward = 0
        # if self.is_success():
        #     goal_reward += 10
        if self.is_failure():
            goal_reward += -5

        def compute_angle(vec1, vec2):
            arr1 = np.array(vec1)
            arr2 = np.array(vec2)
            cosangle = arr1.dot(arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
            angle = min(np.pi / 2, np.abs(np.arccos(cosangle)))
            return angle

        def compute_cirle(point1, vec1, point2, vec2):
            if abs(vec1[1]) < 1e-4:
                vec1[1] = 1e-4
            if abs(vec2[1]) < 1e-4:
                vec2[1] = 1e-4
            k1 = -vec1[0] / (np.sign(vec1[1]) * max(abs(vec1[1]), 1e-4))
            k2 = -vec2[0] / (np.sign(vec2[1]) * max(abs(vec2[1]), 1e-4))
            x1 = point1[0]
            y1 = point1[1]
            x2 = point2[0]
            y2 = point2[1]
            x = (k1 * x1 - k2 * x2 + y2 - y1) / (k1 - k2)
            y = k1 * (x - x1) + y1
            r = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
            return x, y, r

        def compute_point_line_dis(point_x, point_y, vec_x, vec_y, point2_x, point2_y):
            b = np.sqrt(vec_x ** 2 + vec_y ** 2)
            a = abs(vec_x * point2_y - vec_y * point2_x - vec_x * point_y + vec_y * point_x)
            return a / b

        def dist(loc1, loc2):
            return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5

        # retrieve information
        location = self._simulator_databuffer['state']['location']
        speed = self._simulator_databuffer['state']['speed'] / 3.6
        acceleration = self._simulator_databuffer['state']['acceleration']
        forward_vector = self._simulator_databuffer['state']['forward_vector']

        lane_location = self._simulator_databuffer['state']['lane_location']
        lane_forward_vector = self._simulator_databuffer['state']['lane_forward']

        command = self._simulator_databuffer['navigation']['command']
        node = self._simulator_databuffer['navigation']['node']
        next_node = self._simulator_databuffer['navigation']['target']
        node_forward = self._simulator_databuffer['navigation']['node_forward']
        next_forward = self._simulator_databuffer['navigation']['target_forward']
        tick = self._simulator_databuffer['information']['tick']

        distance = self._simulator._planner.distance_to_goal
        # if self.last_distance is None:
        #     dis_reward = 0
        # else:
        #     dis_reward = (self.last_distance - distance) * 1.5
        # self.last_distance = distance

        # if dis_reward == 0 and self.last_location is not None:
        #     dis_reward = min(self.last_location.distance(node_location) - location.distance(node_location), 1.0) * 1.5
        # self.last_location = location
        if dist(node, location) > 3:
            self._dis_reward = -2
        else:
            self._dis_reward = 0

        self._dead_reward = 0
        eq3v = lambda x, y: x[0] == y[0] and x[1] == y[1]
        if eq3v(self._previous_node, node):
            self._stuck_count += 1
            if self._stuck_count >= self._stuck_thres and tick > 30:
                self._dead_reward = -1
        else:
            self._stuck_count = 0
        self.node = node

        # speed reward
        # command {1: 'Left', 2: 'Right', 3: 'Straight', 4: 'Follow'}
        # speed is in m/s in carla
        speed_reward = 0
        speed_limit = self._simulator_databuffer['navigation']['speed_limit']  # km/h here (different from velocity)
        agent_state = self._simulator_databuffer['navigation']['agent_state']

        stuck_reward = self._low_speed_count * -0.1

        # if velocity >= speed_limit:
        #     speed_reward = -1 * (velocity - speed_limit)
        # else:
        #     speed_reward = 0
        speed_reward += min(speed, 5) * 0.2

        if abs(node[0] - next_node[0]) > 1e-1 and abs(node[1] - next_node[1]) > 1e-1:
            new2_flag = True
            cx, cy, r = compute_cirle(node, node_forward, next_node, next_forward)
        else:
            new2_flag = False

        # lane_angle = compute_angle(forward_vector, lane_forward_vector)
        if tick < 30:
            lane_angle = compute_angle(forward_vector, lane_forward_vector)
            hero_lane_distance = dist(node, lane_location)
        else:
            if new2_flag:
                vec1 = np.array([cy - location[1], location[0] - cx])
                angle1 = compute_angle(forward_vector, vec1)
                vec2 = np.array([location[1] - cy, cx - location[0]])
                angle2 = compute_angle(forward_vector, vec2)
                lane_angle = min(angle1, angle2)
                hero_lane_distance = abs(np.sqrt((cx - location[0]) ** 2 + (cy - location[1]) ** 2) - r)
            else:
                lane_angle = compute_angle(forward_vector, node_forward)
                hero_lane_distance = compute_point_line_dis(
                    node[0], node[1], node_forward[0], node_forward[1], location[0], location[1],
                )

        if lane_angle < np.pi / 2:
            lane_angle = lane_angle * 2 / np.pi
            angle_reward = (1 - lane_angle) / (1 + lane_angle) - 1
            angle_reward = max(min(angle_reward, 0), -1)
        else:
            angle_reward = -1

        # hero_lane_distance = max(min(hero_lane_distance, 2), 0)
        # hero_lane_distance = hero_lane_distance / 2.0
        # lane_reward = (1 - hero_lane_distance) / (1 + hero_lane_distance) - 1
        lane_reward = -hero_lane_distance / 3

        # if test_waypoint is None:
        #     angle_reward = -1
        #     lane_reward = -1

        # if command == 1 or command == 2: # go left or go right
        #     dis_reward *= 3.0

        # total_reward = goal_reward + dis_reward + dead_reward + angle_reward + lane_reward + speed_reward
        total_reward = goal_reward + speed_reward + self._dead_reward + stuck_reward + self._dis_reward
        # if lane_reward < -0.99 or angle_reward < -0.99:
        #     total_reward -= 1

        if speed < 0.2 and self._tick > 20:
            self._low_speed_count += 1
        else:
            self._low_speed_count = 0

        info = {}
        info['goal_reward'] = goal_reward
        info['dis_reward'] = self._dis_reward
        info['dead_reward'] = self._dead_reward
        info['angle_reward'] = angle_reward
        info['lane_reward'] = lane_reward
        info['speed_reward'] = speed_reward
        info['hero_lane_distance'] = hero_lane_distance
        info['lane_angle'] = lane_angle / np.pi

        return total_reward, info


class CarlaLatentEvalEnv(CarlaLatentRLEnv):

    def reset(self, col_is_failure=True, **kwargs) -> Dict:
        self._low_speed_count = 0
        self._stuck_count = 0
        return super().reset(col_is_failure=col_is_failure, **kwargs)
