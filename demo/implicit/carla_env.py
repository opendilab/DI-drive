import time
import os
import copy
from enum import Enum
from pathlib import Path
from collections import deque, namedtuple, OrderedDict

import numpy as np
import carla
import torch

from core.simulators import CarlaSimulator
from core.envs.base_drive_env import BaseDriveEnv
from models import ImplicitSupervisedModel
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_tensor
from utils import adapt_order, compute_angle, compute_point_line_dis, compute_cirle, compute_speed

Orders = Enum("Order", "Follow_Lane Straight Right Left ChangelaneLeft ChangelaneRight")


class ImplicitCarlaEnv(BaseDriveEnv):
    '''
    The Carla environment for RL training
    '''

    def __init__(self, args):
        self.args = args
        self.port = args.port
        self.host = args.host
        self._simulator_cfg = self.args.simulator
        self.weather_list = self.args.weather_list
        self.town = self.args.town

        self.n_vehicles = self.args.n_vehicles
        self.n_pedestrians = self.args.n_pedestrians
        self._col_is_failure = True
        self.steps_image = [-10, -2, -1, 0]
        self._simulator = None
        self._launched_simulator = False
        self._init_carla_simulator()

        self.steps = 0
        self._tick = 0
        self._launched_simulator = False
        self.stop_steps = 0
        self.seed()

        if self.args.crop_sky:
            for obs_item in self._simulator_cfg.obs:
                obs_item.update({'fov': 100})
            model_supervised = ImplicitSupervisedModel(
                len(self.steps_image), len(self.steps_image), 1024, 6, 4, self.args.crop_sky
            )
            model_supervised.load_state_dict(torch.load(self.args.supervised_model_path))
        else:
            for obs_item in self._simulator_cfg.obs:
                obs_item.update({'fov': 90})
            model_supervised = ImplicitSupervisedModel(
                len(self.steps_image), len(self.steps_image), 1024, 3, 4, self.args.crop_sky
            )
            model_supervised.load_state_dict(torch.load(self.args.supervised_model_path))
        model_supervised.eval()

        self.encoder = model_supervised.encoder.cuda()
        self.last_conv_downsample = model_supervised.last_conv_downsample.cuda()

        self.action_space = (self.args.nb_action_throttle + 1) * self.args.nb_action_steering

        self.window = (max([abs(number) for number in self.steps_image]) + 1)  # Number of frames to concatenate
        self.State = namedtuple("State", ("image", "speed", "order", "steering"))

        self.RGB_image_buffer = deque([], maxlen=self.window)
        self.state_buffer = deque([], maxlen=self.window)

        if self.args.crop_sky:
            blank_state = self.State(
                np.zeros(6144, dtype=np.float32), -1, -1, 0
            )  # RGB Image, color channet first for torch
        else:
            blank_state = self.State(np.zeros(8192, dtype=np.float32), -1, -1, 0)
        for _ in range(self.window):
            self.state_buffer.append(blank_state)
            if self.args.crop_sky:
                self.RGB_image_buffer.append(
                    np.zeros((3, self.args.front_camera_height - 120, self.args.front_camera_width))
                )
            else:
                self.RGB_image_buffer.append(np.zeros((3, self.args.front_camera_height, self.args.front_camera_width)))

        self.last_steering = 0
        self.last_order = 0

    def _init_carla_simulator(self):
        print(self.host, self.port)
        self._simulator = CarlaSimulator(
            cfg=self._simulator_cfg, client=None, host=self.host, port=self.port
        )
        self._launched_simulator = True

    def seed(self, seed=None):
        np.random.seed(seed)

    def _action_to_control(self, action):
        steer = ((action % self.args.nb_action_steering) - int(self.args.nb_action_steering / 2)
                 ) * (self.args.max_steering / int(self.args.nb_action_steering / 2))
        if action < int(self.args.nb_action_steering * self.args.nb_action_throttle):
            throttle = (int(action / self.args.nb_action_steering
                            )) * (self.args.max_throttle / (self.args.nb_action_throttle - 1))
            brake = 0
        else:
            throttle = 0
            brake = 1.0

        steer = steer
        throttle = throttle
        if brake < 1.0 / 2:
            brake = 0

        control = {}
        control['steer'] = np.clip(steer, -1.0, 1.0)
        control['throttle'] = np.clip(throttle, 0.0, 1.0)
        control['brake'] = np.clip(brake, 0.0, 1.0)
        self.last_steering = steer
        return control

    def step(self, action):
        action = action.item()
        control = self._action_to_control(action)
        self._simulator.apply_control(control)
        self._simulator.run_step()
        self.new_obs = self._get_obs()

        reward, reward_info, distance, forward_vector, velocity = self.compute_reward(self.new_obs, control, self.steps)

        done = self.is_success() or self.is_failure()
        if velocity < 0.2 and (self.new_obs['tl_state'] > 1.5 or self.new_obs['tl_dis'] > 23):
            self.stop_steps += 1
        if velocity > 0.3:
            self.stop_steps = max(0, self.stop_steps - 100)

        if self.new_obs['is_junction'] and velocity > 3.0:
            reward = reward + 0.1

        if reward_info['lane_reward'] < -0.99 or reward_info['angle_reward'] < -0.99:
            done = True
            reward = reward - 1
            print("Step: %d, Terminate! (Offlane)" % self.steps)

        if self.stop_steps > 50 and self.new_obs['tl_state'] > 1.5 and control['throttle'] < 0.1:
            done = True
            reward = reward - 1
            print("Stop: %d, Terminate (Stop)!" % self.steps)

        if self.is_success():
            print("Step: %d, Success !!" % self.steps)

        if self.is_failure():
            print("Step: %d, failure" % self.steps)
            reward = reward - 1

        if self.steps % 20 == 0:
            print(
                "Step: %d, reward: %f, distance: %f, speed: %f, forward: (%f, %f)" %
                (self.steps, reward, distance, velocity, forward_vector.x, forward_vector.y)
            )
            print("Steer: %f, Throttle: %f, Brake: %f" % (control['steer'], control['throttle'], control['brake']))
            strs = []
            for key in reward_info:
                strs.append("%s: %f " % (key, reward_info[key]))
            reward_info_str = ','.join(strs)
            print(reward_info_str)
            print('-' * 80)

        self.steps += 1
        self._tick += 1
        self._final_eval_reward += reward
        info = {}
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        reward = to_tensor([reward]).float()
        obs = {}
        for key in ['speed', 'steer', 'image', 'targets', 'order']:
            obs[key] = self.new_obs[key]
        obs = to_ndarray(obs)
        return BaseEnvTimestep(obs, reward, done, info)

    def info(self):
        return {}

    def __repr__(self):
        return "Carla Env"

    def _get_obs(self):
        observations = self._simulator.get_sensor_data()
        location = self._simulator.hero_player.get_location()
        waypoint = self._simulator._map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        transform = self._simulator.hero_player.get_transform()
        forward_vector = transform.rotation.get_forward_vector()
        state = self._simulator.get_state()
        observations['tl_dis'] = np.float32(state['tl_dis'])
        observations['tl_state'] = state['tl_state']

        right_vec = [forward_vector.y, -forward_vector.x]

        navigation_buffer = self._simulator.get_navigation()
        for key in navigation_buffer:
            observations[key] = navigation_buffer[key]

        observations['is_junction'] = waypoint.is_junction

        node = observations['node']
        next = observations['target']
        node_forward = observations['node_forward']
        next_forward = observations['target_forward']

        targets = []
        targets.append(np.sqrt((location.x - node[0]) ** 2 + (location.y - node[1]) ** 2))
        targets.append(np.sqrt((location.x - next[0]) ** 2 + (location.y - next[1]) ** 2))

        node_sign = np.sign(node_forward[0] * right_vec[0] + node_forward[1] * right_vec[1])
        next_sign = np.sign(next_forward[0] * right_vec[0] + next_forward[1] * right_vec[1])
        targets.append(node_sign * compute_angle([forward_vector.x, forward_vector.y], node_forward))
        targets.append(next_sign * compute_angle([forward_vector.x, forward_vector.y], next_forward))

        rgb = observations["rgb"].copy()
        if self.args.crop_sky:
            rgb = np.array(rgb)[120:, :, :]
        else:
            rgb = np.array(rgb)

        rgb = np.rollaxis(rgb, 2, 0)
        self.RGB_image_buffer.append(rgb)

        #speed = np.linalg.norm(observations["velocity"])
        speed = compute_speed(self._simulator.hero_player.get_velocity())

        order = adapt_order(int(observations["command"]))
        if self.last_order != order:
            print("order = ", Orders(order).name)
            self.last_order = order

        np_array_RGB_input = np.concatenate(
            [self.RGB_image_buffer[indice_image + self.window - 1] for indice_image in self.steps_image]
        )
        torch_tensor_input = (
            torch.from_numpy(np_array_RGB_input).to(dtype=torch.float32).div_(255).unsqueeze(0).cuda()
        )

        with torch.no_grad():
            current_encoding = self.encoder(torch_tensor_input)
            current_encoding = self.last_conv_downsample(current_encoding)

        current_encoding_np = current_encoding.cpu().numpy().flatten()

        current_state = self.State(current_encoding_np, speed, order, self.last_steering)
        self.state_buffer.append(current_state)

        speeds = []
        steerings = []
        for step_image in [-10, -2, -1, 0]:
            state = self.state_buffer[step_image + 10]
            speeds.append(state.speed)
            steerings.append(state.steering)

        observations['command'] = int(observations['command'])
        observations['speed'] = np.array(speeds).astype(np.float32)
        observations['targets'] = np.array(targets).astype(np.float32)
        observations['steer'] = np.array(steerings).astype(np.float32)
        observations['image'] = current_encoding_np.astype(np.float32)
        observations['order'] = current_state.order
        observations = copy.deepcopy(observations)
        del observations['waypoint_list']
        return observations

    def get_changelane_target_waypoint(self, next_loc, next_ori):
        target_loc = []
        target_loc.append(next_loc[0] + next_ori[0] * 10)
        target_loc.append(next_loc[1] + next_ori[1] * 10)
        return target_loc

    def compute_reward(self, ob, control, steps):
        # retrieve information
        location = self._simulator.hero_player.get_location()
        velocity = compute_speed(self._simulator.hero_player.get_velocity())
        acceleration = self._simulator.hero_player.get_acceleration()
        transform = self._simulator.hero_player.get_transform()

        distance = self._simulator._planner.distance_to_goal
        # Rotation: (pitch, yaw, roll) (Y-rotation,Z-rotation,X-rotation)
        forward_vector = transform.rotation.get_forward_vector()
        carla_map = self._simulator._map

        waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

        test_waypoint = carla_map.get_waypoint(location, project_to_road=True)
        if test_waypoint is None:
            hero_out_lane = True
        else:
            hero_out_lane = False

        # speed reward
        # command {1: 'Left', 2: 'Right', 3: 'Straight', 4: 'Follow'}
        # speed is in m/s in carla
        command = ob['command']

        tl_state = ob['tl_state']
        tl_dis = ob['tl_dis']
        is_junction = ob['is_junction']

        node = ob['node'].astype(np.float64)
        next = ob['target'].astype(np.float64)
        node_forward = ob['node_forward'].astype(np.float64)
        next_forward = ob['target_forward'].astype(np.float64)

        node_location = carla.Location(node[0], node[1], 0)
        node_forward_vector = carla.Vector3D(x=node_forward[0], y=node_forward[1], z=0)

        if is_junction:
            target_speed = 5
        elif tl_state == 2 or tl_state == 3:
            target_speed = 5
        elif tl_dis > 23:
            target_speed = 5
        elif tl_dis < 8:
            target_speed = 0
        else:
            target_speed = (tl_dis - 8) / 3

        speed_reward = max(0, 1 - 0.2 * np.abs(velocity - target_speed))

        k1 = -node_forward[0] / (np.sign(node_forward[1]) * max(abs(node_forward[1]), 1e-3))
        k2 = -next_forward[0] / (np.sign(next_forward[1]) * max(abs(next_forward[1]), 1e-3))

        # Judge whether the vehicle is turning. If so, we will calculate the reward in another way
        if command == 5 or command == 6:
            changelane_flag = True
            new2_flag = False
        if abs(node[0] - next[0]) > 1e-1 and abs(node[1] - next[1]) > 1e-1 and np.abs(k1 - k2) > 1e-1:
            circle_flag = True
            changelane_flag = False
            cx, cy, r = compute_cirle(node, node_forward, next, next_forward)
        else:
            circle_flag = False
            changelane_flag = False

        # angle reward
        if hero_out_lane:
            angle_reward = -1
            # lane distance reward
            # only choose corners and center to calcuate
            lane_reward = -1
        else:
            lane_forward_vector = waypoint.transform.rotation.get_forward_vector()
            waypoint_location = waypoint.transform.location
            # the Carla simulator exists some bug at the beginning, so we use the easy method to compute reward
            if steps < 30:
                lane_angle = compute_angle(forward_vector, lane_forward_vector)
            else:
                if circle_flag:
                    angle1 = compute_angle(
                        forward_vector, carla.Vector3D(x=(cy - location.y), y=(location.x - cx), z=0)
                    )
                    angle2 = compute_angle(
                        forward_vector, carla.Vector3D(x=(location.y - cy), y=(cx - location.x), z=0)
                    )
                    lane_angle = min(angle1, angle2)
                elif changelane_flag:
                    if command == 5:
                        target_loc = self.get_changelane_target_waypoint(next, next_forward)
                    if command == 6:
                        target_loc = self.get_changelane_target_waypoint(next, next_forward)
                else:
                    lane_angle = compute_angle(forward_vector, node_forward_vector)
            if lane_angle < np.pi / 2:
                lane_angle = lane_angle * 2 / np.pi
                angle_reward = (1 - lane_angle) / (1 + lane_angle) - 1
                angle_reward = max(min(angle_reward, 0), -1)
            else:
                angle_reward = -1

            if steps < 30:
                hero_lane_distance = location.distance(waypoint_location)
            else:
                if circle_flag:
                    hero_lane_distance = abs(np.sqrt((cx - location.x) ** 2 + (cy - location.y) ** 2) - r)
                elif changelane_flag:
                    if command == 5:
                        target_loc = self.get_changelane_target_waypoint(next, next_forward)
                    if command == 6:
                        target_loc = self.get_changelane_target_waypoint(next, next_forward)
                    hero_lane_distance = compute_point_line_dis(
                        node_location.x, node_location.y, target_loc[0] - node[0], target_loc[1] - node[1], location.x,
                        location.y
                    )
                else:
                    hero_lane_distance = compute_point_line_dis(
                        node_location.x, node_location.y, node_forward_vector.x, node_forward_vector.y, location.x,
                        location.y
                    )
            hero_lane_distance = max(min(hero_lane_distance, 2), 0)
            hero_lane_distance = hero_lane_distance / 2.0
            lane_reward = (1 - hero_lane_distance) / (1 + hero_lane_distance) - 1

        total_reward = speed_reward + angle_reward + lane_reward

        info = {}
        info['speed_reward'] = speed_reward
        info['angle_reward'] = angle_reward
        info['lane_reward'] = lane_reward

        return total_reward, info, distance, forward_vector, velocity

    def reset(self):
        self.steps = 0
        self.stop_steps = 0
        self._tick = 0

        self.RGB_image_buffer = deque([], maxlen=self.window)
        self.state_buffer = deque([], maxlen=self.window)

        if self.args.crop_sky:
            blank_state = self.State(
                np.zeros(6144, dtype=np.float32), -1, -1, 0
            )  # RGB Image, color channet first for torch
        else:
            blank_state = self.State(np.zeros(8192, dtype=np.float32), -1, -1, 0)
        for _ in range(self.window):
            self.state_buffer.append(blank_state)
            if self.args.crop_sky:
                self.RGB_image_buffer.append(
                    np.zeros((3, self.args.front_camera_height - 120, self.args.front_camera_width))
                )
            else:
                self.RGB_image_buffer.append(np.zeros((3, self.args.front_camera_height, self.args.front_camera_width)))
        self.last_steering = 0
        self.last_order = 0
        np.random.seed(int(time.time()))

        start = np.random.randint(240)
        target = np.random.randint(240)
        while abs(target - start) < 5:
            target = np.random.randint(240)

        env_params = {
            'weather': np.random.choice(self.weather_list),
            'start': start,
            'target': target,
            'town': self.town,
            'n_pedestrians': self.n_pedestrians,
            'n_vehicles': self.n_vehicles,
        }

        self._simulator.init(**env_params)
        self._simulator.run_step()
        self._success_dis = 5.0
        self._final_eval_reward = 0.
        self._timeout = self._simulator._planner.timeout
        self.new_obs = self._get_obs()
        obs = {}
        for key in ['speed', 'steer', 'image', 'targets', 'order']:
            obs[key] = self.new_obs[key]
        obs = to_ndarray(obs)
        return obs

    def is_failure(self):
        if self._tick >= self._timeout:
            return True
        elif self._col_is_failure and self._simulator.collided:
            return True
        return False

    def is_success(self):
        return self._simulator._planner.distance_to_goal < self._success_dis

    def close(self):
        if self._launched_simulator:
            self._simulator.__exit__()
            del self._simulator
            self._launched_simulator = False
