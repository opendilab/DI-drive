from collections import namedtuple
import os
from ding.torch_utils.data_helper import to_device, to_dtype, to_tensor
import torch
from torchvision import transforms
import numpy as np
from typing import Dict, List, Any, Optional

from .base_carla_policy import BaseCarlaPolicy
from core.models import PIDController, CustomController
from core.models.lbc_model import LBCBirdviewModel, LBCImageModel
from core.utils.model_utils import common
from ding.utils.data import default_collate, default_decollate
from core.utils.learner_utils.loss_utils import LocationLoss

STEPS = 5
SPEED_STEPS = 3
COMMANDS = 4


class LBCBirdviewPolicy(BaseCarlaPolicy):
    """
    LBC driving policy with Bird-eye View inputs. It has an LBC NN model which can handle
    observations from several environments by collating data into batch. Each environment
    has a PID controller related to it to get final control signals. In each updating, all
    envs should use the correct env id to make the PID controller works well, and the
    controller should be reset when starting a new episode.

    :Arguments:
        - cfg (Dict): Config Dict.

    :Interfaces:
        reset, forward
    """

    config = dict(
        cuda=True,
        model=dict(),
        learn=dict(loss='l1', ),
        steer_points=None,
        pid=None,
        gap=5,
        dt=0.1,
        crop_size=192,
        pixels_per_meter=5,
    )

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg, enable_field=[])
        self._enable_field = ['eval', 'learn']
        self._controller_dict = dict()

        if self._cfg.cuda:
            if not torch.cuda.is_available():
                print('[POLICY] No cuda device found! Use cpu by default')
                self._device = torch.device('cpu')
            else:
                self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self._one_hot = torch.FloatTensor(torch.eye(4))
        self._transform = transforms.ToTensor()

        self._gap = self._cfg.gap
        self._dt = self._cfg.dt
        self._crop_size = self._cfg.crop_size
        self._pixels_per_meter = self._cfg.pixels_per_meter

        self._steer_points = self._cfg.steer_points
        self._pid = self._cfg.pid
        if self._steer_points is None:
            self._steer_points = {"1": 3, "2": 2, "3": 2, "4": 2}

        if self._pid is None:
            self._pid = {
                "1": {
                    "Kp": 1.0,
                    "Ki": 0.1,
                    "Kd": 0
                },  # Left
                "2": {
                    "Kp": 1.0,
                    "Ki": 0.1,
                    "Kd": 0
                },  # Right
                "3": {
                    "Kp": 0.8,
                    "Ki": 0.1,
                    "Kd": 0
                },  # Straight
                "4": {
                    "Kp": 0.8,
                    "Ki": 0.1,
                    "Kd": 0
                },  # Follow
            }

        self._speed_control_func = lambda: PIDController(K_P=1.0, K_I=0.1, K_D=2.5)
        self._turn_control_func = lambda: CustomController(self._pid)

        self._model = LBCBirdviewModel(**self._cfg.model)
        self._model.to(self._device)

        for field in self._enable_field:
            getattr(self, '_init_' + field)()

    def _init_learn(self) -> None:
        if self._cfg.learn.loss == 'l1':
            self._criterion = LocationLoss(choice='l1')
        elif self._cfg.policy.learn.loss == 'l2':
            self._criterion = LocationLoss(choice='l2')

    def _postprocess(self, steer, throttle, brake):
        control = {}
        control.update(
            {
                'steer': np.clip(steer, -1.0, 1.0),
                'throttle': np.clip(throttle, 0.0, 1.0),
                'brake': np.clip(brake, 0.0, 1.0),
            }
        )
        return control

    def _reset_single(self, data_id):
        if data_id in self._controller_dict:
            self._controller_dict.pop(data_id)

        self._controller_dict[data_id] = (self._speed_control_func(), self._turn_control_func())

    def _reset(self, data_ids: Optional[List[int]] = None) -> None:
        if data_ids is not None:
            for id in data_ids:
                self._reset_single(id)
        else:
            for id in self._controller_dict:
                self._reset_single(id)

    def _forward_eval(self, data: Dict) -> Dict[str, Any]:
        """
        Running forward to get control signal of `eval` mode.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: Control and waypoints dict stored in values for each provided env id.
        """

        data_ids = list(data.keys())

        data = default_collate(list(data.values()))

        birdview = to_dtype(data['birdview'], dtype=torch.float32).permute(0, 3, 1, 2)
        speed = data['speed']
        command_index = [i.item() - 1 for i in data['command']]
        command = self._one_hot[command_index]
        if command.ndim == 1:
            command = command.unsqueeze(0)

        with torch.no_grad():
            _birdview = birdview.to(self._device)
            _speed = speed.to(self._device)
            _command = command.to(self._device)

            if self._model._all_branch:
                _locations, _ = self._model(_birdview, _speed, _command)
            else:
                _locations = self._model(_birdview, _speed, _command)
            _locations = _locations.detach().cpu().numpy()

        map_locations = _locations
        actions = {}

        for index, data_id in enumerate(data_ids):
            # Pixel coordinates.
            map_location = map_locations[index, ...]
            map_location = (map_location + 1) / 2 * self._crop_size

            targets = list()

            for i in range(STEPS):
                pixel_dx, pixel_dy = map_location[i]
                pixel_dx = pixel_dx - self._crop_size / 2
                pixel_dy = self._crop_size - pixel_dy

                angle = np.arctan2(pixel_dx, pixel_dy)
                dist = np.linalg.norm([pixel_dx, pixel_dy]) / self._pixels_per_meter

                targets.append([dist * np.cos(angle), dist * np.sin(angle)])

            target_speed = 0.0

            for i in range(1, SPEED_STEPS):
                pixel_dx, pixel_dy = map_location[i]
                prev_dx, prev_dy = map_location[i - 1]

                dx = pixel_dx - prev_dx
                dy = pixel_dy - prev_dy
                delta = np.linalg.norm([dx, dy])

                target_speed += delta / (self._pixels_per_meter * self._gap * self._dt) / (SPEED_STEPS - 1)

            _cmd = data['command'][index].item()
            _sp = data['speed'][index].item()
            n = self._steer_points.get(str(_cmd), 1)
            targets = np.concatenate([[[0, 0]], targets], 0)
            c, r = ls_circle(targets)
            closest = common.project_point_to_circle(targets[n], c, r)

            v = [1.0, 0.0, 0.0]
            w = [closest[0], closest[1], 0.0]
            alpha = common.signed_angle(v, w)
            steer = self._controller_dict[data_id][1].run_step(alpha, _cmd)
            throttle = self._controller_dict[data_id][0].step(target_speed - _sp)
            brake = 0.0

            if target_speed < 1.0:
                steer = 0.0
                throttle = 0.0
                brake = 1.0

            control = self._postprocess(steer, throttle, brake)
            control.update({'map_locations': map_location})
            actions[data_id] = {'action': control}
        return actions

    def _reset_eval(self, data_ids: Optional[List[int]] = None) -> None:
        """
        Reset policy of `eval` mode. It will change the NN model into 'eval' mode and reset
        the controllers in providded env id.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        self._model.eval()
        self._reset(data_ids)

    def _forward_learn(self, data: Dict) -> Dict[str, Any]:
        birdview = to_dtype(data['birdview'], dtype=torch.float32).permute(0, 3, 1, 2)
        speed = to_dtype(data['speed'], dtype=torch.float32)
        command_index = [i.item() - 1 for i in data['command']]
        command = self._one_hot[command_index]
        if command.ndim == 1:
            command = command.unsqueeze(0)

        _birdview = birdview.to(self._device)
        _speed = speed.to(self._device)
        _command = command.to(self._device)

        if self._model._all_branch:
            _locations, _ = self._model(_birdview, _speed, _command)
        else:
            _locations = self._model(_birdview, _speed, _command)

        locations_pred = _locations
        location_gt = data['location'].to(self._device)
        loss = self._criterion(locations_pred, location_gt)
        return {
            'loss': loss,
            'locations_pred': locations_pred,
        }

    def _reset_learn(self, data_ids: Optional[List[int]] = None) -> None:
        self._model.train()


class LBCImagePolicy(BaseCarlaPolicy):
    """
    LBC driving policy with RGB image inputs. It has an LBC NN model which can handle
    observations from several environments by collating data into batch. Each environment
    has a PID controller related to it to get final control signals. In each updating, all
    envs should use the correct env id to make the PID controller works well, and the
    controller should be reset when starting a new episode.

    :Arguments:
        - cfg (Dict): Config Dict.

    :Interfaces:
        reset, forward
    """

    config = dict(
        model=dict(cuda=True, backbone='resnet34', all_branch=False),
        camera_args=dict(
            fixed_offset=4.0,
            fov=90,
            h=160,
            w=384,
            world_y=1.4,
        ),
        steer_points=None,
        pid=None,
        gap=5,
        dt=0.1,
    )

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg, enable_field=set(['eval', 'learn']))
        self._controller_dict = dict()

        if self._cfg.model.cuda:
            if not torch.cuda.is_available():
                print('[POLICY] No cuda device found! Use cpu by default')
                self._device = torch.device('cpu')
            else:
                self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self._one_hot = torch.FloatTensor(torch.eye(4))
        self._transform = transforms.ToTensor()

        self._camera_args = self._cfg.camera_args
        self._fixed_offset = self._camera_args.fixed_offset
        w = float(self._camera_args.w)
        h = float(self._camera_args.h)
        self._img_size = np.array([w, h])

        self._gap = self._cfg.gap
        self._dt = self._cfg.dt

        self._steer_points = self._cfg.steer_points
        self._pid = self._cfg.pid
        if self._steer_points is None:
            self._steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}

        if self._pid is None:
            self._pid = {
                "1": {
                    "Kp": 0.5,
                    "Ki": 0.20,
                    "Kd": 0.0
                },
                "2": {
                    "Kp": 0.7,
                    "Ki": 0.10,
                    "Kd": 0.0
                },
                "3": {
                    "Kp": 1.0,
                    "Ki": 0.10,
                    "Kd": 0.0
                },
                "4": {
                    "Kp": 1.0,
                    "Ki": 0.50,
                    "Kd": 0.0
                }
            }

        self._speed_control_func = lambda: PIDController(K_P=.8, K_I=.08, K_D=0.)
        self._turn_control_func = lambda: CustomController(self._pid)

        self._engine_brake_threshold = 2.0
        self._brake_threshold = 2.0

        self._model = LBCImageModel(self._cfg.model.backbone, False, all_branch=self._cfg.model.all_branch)
        self._model = self._model.to(self._device)

    def _init_learn(self) -> None:
        if self._cfg.learn.loss == 'l1':
            self._criterion = LocationLoss(choise='l1')
        elif self._cfg.policy.learn.loss == 'l2':
            self._criterion = LocationLoss(choise='l2')

    def _reset_single(self, data_id):
        if data_id in self._controller_dict:
            self._controller_dict.pop(data_id)

        self._controller_dict[data_id] = (self._speed_control_func(), self._turn_control_func())

    def _reset(self, data_ids: Optional[List[int]] = None) -> None:
        if data_ids is not None:
            for id in data_ids:
                self._reset_single(id)
        else:
            for id in self._controller_dict:
                self._reset_single(id)

    def _postprocess(self, steer, throttle, brake):
        control = {}
        control.update(
            {
                'steer': np.clip(steer, -1.0, 1.0),
                'throttle': np.clip(throttle, 0.0, 1.0),
                'brake': np.clip(brake, 0.0, 1.0),
            }
        )

        return control

    def _unproject(self, output, world_y=1.4, fov=90):

        cx, cy = self._img_size / 2

        w, h = self._img_size

        f = w / (2 * np.tan(fov * np.pi / 360))

        xt = (output[..., 0:1] - cx) / f
        yt = (output[..., 1:2] - cy) / f

        world_z = world_y / yt
        world_x = world_z * xt

        world_output = np.stack([world_x, world_z], axis=-1)

        if self._fixed_offset:
            world_output[..., 1] -= self._fixed_offset

        world_output = world_output.squeeze()

        return world_output

    def _forward_eval(self, data: Dict) -> Dict:
        """
        Running forward to get control signal of `eval` mode.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: Control and waypoints dict stored in values for each provided env id.
        """

        data_ids = list(data.keys())

        data = default_collate(list(data.values()))

        rgb = to_dtype(data['rgb'], dtype=torch.float32).permute(0, 3, 1, 2)
        speed = data['speed']
        command_index = [i.item() - 1 for i in data['command']]
        command = self._one_hot[command_index]
        if command.ndim == 1:
            command = command.unsqueeze(0)

        with torch.no_grad():
            _rgb = rgb.to(self._device)
            _speed = speed.to(self._device)
            _command = command.to(self._device)
            if self._model.all_branch:
                model_pred, _ = self._model(_rgb, _speed, _command)
            else:
                model_pred = self._model(_rgb, _speed, _command)

        model_pred = model_pred.detach().cpu().numpy()

        pixels_pred = model_pred
        actions = {}

        for index, data_id in enumerate(data_ids):

            # Project back to world coordinate
            pixel_pred = pixels_pred[index, ...]
            pixel_pred = (pixel_pred + 1) * self._img_size / 2

            world_pred = self._unproject(pixel_pred, self._camera_args.world_y, self._camera_args.fov)

            targets = [(0, 0)]

            for i in range(STEPS):
                pixel_dx, pixel_dy = world_pred[i]
                angle = np.arctan2(pixel_dx, pixel_dy)
                dist = np.linalg.norm([pixel_dx, pixel_dy])

                targets.append([dist * np.cos(angle), dist * np.sin(angle)])

            targets = np.array(targets)
            target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (self._gap * self._dt)

            _cmd = data['command'][index].item()
            _sp = data['speed'][index].item()

            c, r = ls_circle(targets)
            n = self._steer_points.get(str(_cmd), 1)
            closest = common.project_point_to_circle(targets[n], c, r)

            v = [1.0, 0.0, 0.0]
            w = [closest[0], closest[1], 0.0]
            alpha = common.signed_angle(v, w)

            steer = self._controller_dict[data_id][1].run_step(alpha, _cmd)
            throttle = self._controller_dict[data_id][0].step(target_speed - _sp)
            brake = 0.0

            # Slow or stop.
            if target_speed <= self._engine_brake_threshold:
                steer = 0.0
                throttle = 0.0

            if target_speed <= self._brake_threshold:
                brake = 1.0

            control = self._postprocess(steer, throttle, brake)
            control.update({'map_locations': pixels_pred})
            actions[data_id] = {'action': control}
        return actions

    def _reset_eval(self, data_ids: Optional[List[int]]) -> None:
        """
        Reset policy of `eval` mode. It will change the NN model into 'eval' mode and reset
        the controllers in providded env id.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        self._model.eval()
        self._reset(data_ids)

    def _forward_learn(self, data: Dict) -> Dict[str, Any]:
        rgb = to_dtype(data['rgb'], dtype=torch.float32).permute(0, 3, 1, 2)
        speed = to_dtype(data['speed'], dtype=torch.float32)
        command_index = [i.item() - 1 for i in data['command']]
        command = self._one_hot[command_index]
        if command.ndim == 1:
            command = command.unsqueeze(0)

        _rgb = rgb.to(self._device)
        _speed = speed.to(self._device)
        _command = command.to(self._device)

        if self._model._all_branch:
            _locations, _ = self._model(_rgb, _speed, _command)
        else:
            _locations = self._model(_rgb, _speed, _command)

        locations_pred = _locations
        location_gt = data['location'].to(self._device)
        loss = self._criterion(locations_pred, location_gt)
        return {
            'loss': loss,
            'location_pred': locations_pred,
        }

    def _reset_learn(self, data_ids: Optional[List[int]] = None) -> None:
        self._model.train()


def ls_circle(points):
    '''
    Input: Nx2 points
    Output: cx, cy, r
    '''
    xs = points[:, 0]
    ys = points[:, 1]

    us = xs - np.mean(xs)
    vs = ys - np.mean(ys)

    Suu = np.sum(us ** 2)
    Suv = np.sum(us * vs)
    Svv = np.sum(vs ** 2)
    Suuu = np.sum(us ** 3)
    Suvv = np.sum(us * vs * vs)
    Svvv = np.sum(vs ** 3)
    Svuu = np.sum(vs * us * us)

    A = np.array([[Suu, Suv], [Suv, Svv]])

    b = np.array([1 / 2. * Suuu + 1 / 2. * Suvv, 1 / 2. * Svvv + 1 / 2. * Svuu])

    cx, cy = np.linalg.solve(A, b)
    r = np.sqrt(cx * cx + cy * cy + (Suu + Svv) / len(xs))

    cx += np.mean(xs)
    cy += np.mean(ys)

    return np.array([cx, cy]), r
