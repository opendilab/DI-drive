import os
import re
import torch
import numpy as np
import PIL
from easydict import EasyDict
from enum import Enum
from collections import deque, namedtuple

from models import ImplicitDQN, ImplicitSupervisedModel
from utils import adapt_order, compute_angle

Orders = Enum("Order", "Follow_Lane Straight Right Left ChangelaneLeft ChangelaneRight")


class ImplicitPolicy():

    def __init__(self, cfg):
        self.args = cfg
        self.args.steps_image = [-10, -2, -1, 0]

        self.window = 11
        self.State = namedtuple("State", ("image", "speed", "order", "steering"))

        self._last_steering_dict = {}  #
        self._last_order_dict = {}
        self._RGB_image_buffer_dict = {}
        self._state_buffer_dict = {}

        path_to_folder_with_model = self.args.path_folder_model
        path_to_model_supervised = os.path.join(path_to_folder_with_model, "model_supervised/")
        path_model_supervised = None
        for file in os.listdir(path_to_model_supervised):
            if ".pth" in file:
                path_model_supervised = os.path.join(path_to_model_supervised, file)
        if path_model_supervised is None:
            raise ValueError("We didn't find any model supervised in folder " + path_to_model_supervised)

        # All this magic number should match the one used when training supervised...
        model_supervised = ImplicitSupervisedModel(4, 4, 1024, 6, 4, self.args.crop_sky)
        model_supervised.load_state_dict(torch.load(path_model_supervised))

        if self.args.multi_lanes:
            model_supervised = model_supervised.eval()

        self.encoder = model_supervised.encoder.cuda()
        self.last_conv_downsample = model_supervised.last_conv_downsample.cuda()

        self.action_space = (self.args.nb_action_throttle + 1) * self.args.nb_action_steering

        path_to_model_RL = os.path.join(path_to_folder_with_model, "model_RL")
        os.chdir(path_to_model_RL)
        tab_model = []
        for file in os.listdir(path_to_model_RL):
            if ".pth" in file:
                tab_model.append(os.path.join(path_to_model_RL, file))

        if len(tab_model) == 0:
            raise ValueError("We didn't find any RL model in folder " + path_to_model_RL)

        self.tab_RL_model = []
        for current_model in tab_model:
            current_RL_model = ImplicitDQN(self.action_space, crop_sky=self.args.crop_sky).cuda()
            current_RL_model_dict = current_RL_model.state_dict()

            print("we load RL model ", current_model)
            checkpoint = torch.load(current_model)
            #checkpoint = checkpoint['model']

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in current_RL_model_dict}
            # 2. overwrite entries in the existing state dict
            current_RL_model_dict.update(pretrained_dict)
            # 3. load the new state dict
            current_RL_model.load_state_dict(current_RL_model_dict)
            self.tab_RL_model.append(current_RL_model)

    def _act(self, targets, state_buffer, RL_model):
        speeds = []
        order = state_buffer[-1].order
        steerings = []
        for step_image in self.args.steps_image:
            state = state_buffer[step_image + self.window - 1]
            speeds.append(state.speed)
            steerings.append(state.steering)
        images = torch.from_numpy(state_buffer[-1].image).float().cuda()
        speeds = torch.from_numpy(np.stack(speeds).astype(np.float32)).float().cuda()
        steerings = torch.from_numpy(np.stack(steerings).astype(np.float32)).float().cuda()
        targets = torch.from_numpy(np.array(targets).astype(np.float32)).float().cuda()
        with torch.no_grad():
            obs = {}
            obs['speed'] = speeds.unsqueeze(0)
            obs['steer'] = steerings.unsqueeze(0)
            obs['targets'] = targets.unsqueeze(0)
            obs['image'] = images.unsqueeze(0)
            obs['order'] = order
            quantile_values = RL_model(obs)
            return quantile_values['logit'].mean(0).argmax(0).item()

    def _reset(self, data_id):
        self._last_steering_dict[data_id] = 0
        self._last_order_dict[data_id] = 0
        self._RGB_image_buffer_dict[data_id] = deque([], maxlen=self.window)
        self._state_buffer_dict[data_id] = deque([], maxlen=self.window)

        if self.args.crop_sky:
            blank_state = self.State(
                np.zeros(6144, dtype=np.float32), -1, -1, 0
            )  # RGB Image, color channet first for torch
        else:
            blank_state = self.State(np.zeros(8192, dtype=np.float32), -1, -1, 0)
        for _ in range(self.window):
            self._state_buffer_dict[data_id].append(blank_state)
            if self.args.crop_sky:
                self._RGB_image_buffer_dict[data_id].append(
                    np.zeros((3, self.args.front_camera_height - 120, self.args.front_camera_width))
                )
            else:
                self._RGB_image_buffer_dict[data_id].append(
                    np.zeros((3, self.args.front_camera_height, self.args.front_camera_width))
                )

    def reset(self, data_id):
        if data_id is not None:
            for id in data_id:
                self._reset(id)
        else:
            for id in self._state_bufffer_dict:
                self._reset(id)

    def forward(self, data):
        actions = dict()
        for i in data.keys():
            obs = data[i]
            action = self.run(i, obs)
            actions[i] = {'action': action}
        return actions

    def run(self, data_id, observations):
        rgb = observations["rgb"].copy()
        if self.args.crop_sky:
            rgb = np.array(rgb)[120:, :, :]
        else:
            rgb = np.array(rgb)

        rgb = np.rollaxis(rgb, 2, 0)
        self._RGB_image_buffer_dict[data_id].append(rgb)

        speed = observations["speed"] / 3.6

        order = adapt_order(int(observations["command"]))
        if self._last_order_dict[data_id] != order:
            print("order = ", Orders(order).name)

            self._last_order_dict[data_id] = order

        np_array_RGB_input = np.concatenate(
            [self._RGB_image_buffer_dict[data_id][indice_image + self.window - 1] for indice_image in [-10, -2, -1, 0]]
        )
        torch_tensor_input = (
            torch.from_numpy(np_array_RGB_input).to(dtype=torch.float32).div_(255).unsqueeze(0).cuda()
        )

        with torch.no_grad():
            current_encoding = self.encoder(torch_tensor_input)

            current_encoding = self.last_conv_downsample(current_encoding)

        current_encoding_np = current_encoding.cpu().numpy().flatten()

        current_state = self.State(current_encoding_np, speed, order, self._last_steering_dict[data_id])
        self._state_buffer_dict[data_id].append(current_state)

        forward_vector = observations['forward_vector']
        location = observations['location']
        node = observations['node']
        next = observations['target']
        node_forward = observations['node_forward']
        next_forward = observations['target_forward']
        right_vec = [forward_vector[1], -forward_vector[0]]

        targets = []
        targets.append(np.sqrt((location[0] - node[0]) ** 2 + (location[1] - node[1]) ** 2))
        targets.append(np.sqrt((location[0] - next[0]) ** 2 + (location[1] - next[1]) ** 2))

        node_sign = np.sign(node_forward[0] * right_vec[0] + node_forward[1] * right_vec[1])
        next_sign = np.sign(next_forward[0] * right_vec[0] + next_forward[1] * right_vec[1])
        targets.append(node_sign * compute_angle([forward_vector[0], forward_vector[1]], node_forward))
        targets.append(next_sign * compute_angle([forward_vector[0], forward_vector[1]], next_forward))

        tab_action = []

        for RL_model in self.tab_RL_model:
            current_action = self._act(targets, self._state_buffer_dict[data_id], RL_model)
            tab_action.append(current_action)

        steer = 0
        throttle = 0
        brake = 0

        for action in tab_action:

            steer += ((action % self.args.nb_action_steering) - int(self.args.nb_action_steering / 2)
                      ) * (self.args.max_steering / int(self.args.nb_action_steering / 2))
            if action < int(self.args.nb_action_steering * self.args.nb_action_throttle):
                throttle += (int(action / self.args.nb_action_steering
                                 )) * (self.args.max_throttle / (self.args.nb_action_throttle - 1))
                brake += 0
            else:
                throttle += 0
                brake += 1.0

        steer = steer / len(tab_action)
        throttle = throttle / len(tab_action)
        if brake < len(tab_action) / 2:
            brake = 0
        else:
            brake = brake / len(tab_action)

        action = {'steer': steer, 'throttle': throttle, 'brake': brake}
        self._last_steering_dict[data_id] = steer
        return action
