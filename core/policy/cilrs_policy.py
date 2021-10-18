from typing import Dict, List, Optional, Any
from collections import namedtuple
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

from ding.utils.data import default_collate, default_decollate
from ding.torch_utils import to_device
from core.models import CILRSModel
from .base_carla_policy import BaseCarlaPolicy


class CILRSPolicy(BaseCarlaPolicy):
    config = dict(
        cuda=True,
        max_throttle=0.75,
        model=dict(),
        learn=dict(
            epoches=200,
            lr=1e-4,
            batch_size=128,
            loss='l1',
            speed_weight=0.05,
            control_weights=[0.5, 0.45, 0.05],
        ),
    )
    validate_function = namedtuple(
        'validate_function', [
            'forward',
            'reset',
            'get_attribute',
            'set_attribute',
            'state_dict',
            'load_state_dict',
        ]
    )

    def __init__(
            self,
            cfg: Dict,
    ) -> None:
        super().__init__(cfg, enable_field=[])
        self._enable_field = ['eval', 'learn', 'validate']
        self._cuda = self._cfg.cuda
        self._max_throttle = self._cfg.max_throttle
        self._model = CILRSModel(**self._cfg.model)
        if self._cuda:
            self._model.cuda()

        for field in self._enable_field:
            getattr(self, '_init_' + field)()

    def _process_sensors(self, sensor):
        sensor = sensor[:, :, ::-1]  # BGR->RGB
        sensor = np.transpose(sensor, (2, 0, 1))
        sensor = sensor / 255.0

        return sensor

    def _process_model_outputs(self, output):

        control_pred, velocity_pred = output
        steer = control_pred[:, 0] * 2 - 1.  # convert from [0,1] to [-1,1]
        throttle = control_pred[:, 1] * self._cfg.max_throttle
        brake = control_pred[:, 2]
        if brake < 0.05:
            brake = brake - brake
        action = {'steer': steer, 'throttle': throttle, 'brake': brake}

        return action, velocity_pred

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._model.eval()

    @torch.no_grad()
    def _forward_eval(self, data: Dict):
        data_id = list(data.keys())

        new_data = dict()
        for id in data.keys():
            new_data[id] = dict()
            new_data[id]['rgb'] = self._process_sensors(data[id]['rgb'].numpy())
            new_data[id]['command'] = data[id]['command']
            new_data[id]['speed'] = data[id]['speed']

        data = default_collate(list(new_data.values()))
        if self._cuda:
            data = to_device(data, 'cuda')

        embedding = self._model.encode([data['rgb']])
        output = self._model(embedding, data['speed'], data['command'])
        if self._cuda:
            output = to_device(output, 'cpu')

        actions, _ = self._process_model_outputs(output)
        actions = default_decollate(actions)
        return {i: {'action': d} for i, d in zip(data_id, actions)}

    def _init_learn(self) -> None:
        self._optimizer = Adam(self._model.parameters(), self._cfg.learn.lr)
        if self._cfg.learn.loss == 'l1':
            self._criterion = F.l1_loss
        elif self._cfg.policy.learn.loss == 'l2':
            self._criterion = F.mse_loss

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._model.train()

    def _forward_learn(self, data: Dict) -> Dict[str, Any]:
        if self._cuda:
            data = to_device(data, 'cuda')

        rgb = data['rgb']
        steer_gt, throttle_gt, brake_gt = data['steer'], data['throttle'], data['brake']
        speed = data['speed']
        command = data['command']
        embedding = self._model.encode([rgb])
        output = self._model(embedding, speed, command)
        control_pred, speed_pred = output
        steer_pred = control_pred[:, 0]
        throttle_pred = control_pred[:, 1]
        brake_pred = control_pred[:, 2]

        speed_loss = self._criterion(speed_pred.squeeze(), speed.squeeze()).mean() * self._cfg.learn.speed_weight
        steer_loss = self._criterion(steer_pred, steer_gt.squeeze()).mean() * self._cfg.learn.control_weights[0]
        throttle_loss = self._criterion(throttle_pred,
                                        throttle_gt.squeeze()).mean() * self._cfg.learn.control_weights[1]
        brake_loss = self._criterion(brake_pred, brake_gt.squeeze()).mean() * self._cfg.learn.control_weights[2]

        total_loss = speed_loss + steer_loss + throttle_loss + brake_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        return_info = {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'speed_loss': speed_loss.item(),
            'steer_loss': steer_loss.item(),
            'throttle_loss': throttle_loss.item(),
            'brake_loss': brake_loss.item(),
            # 'steer_mean': steer_pred.item().mean(),
            # 'throttle_mean': throttle_pred.item().mean(),
            # 'brake_mean': brake_pred.item().mean(),
        }

        return return_info

    def _init_validate(self):
        if self._cfg.learn.loss == 'l1':
            self._criterion = F.l1_loss
        elif self._cfg.policy.learn.loss == 'l2':
            self._criterion = F.mse_loss

    @torch.no_grad()
    def _forward_validate(self, data: Dict) -> Dict[str, Any]:
        if self._cuda:
            data = to_device(data, 'cuda')

        rgb = data['rgb']
        steer_gt, throttle_gt, brake_gt = data['steer'], data['throttle'], data['brake']
        speed = data['speed']
        command = data['command']
        embedding = self._model.encode([rgb])
        output = self._model(embedding, speed, command)
        control_pred, speed_pred = output
        steer_pred = control_pred[:, 0]
        throttle_pred = control_pred[:, 1]
        brake_pred = control_pred[:, 2]

        speed_loss = self._criterion(speed_pred.squeeze(), speed.squeeze()).mean() * self._cfg.learn.speed_weight
        steer_loss = self._criterion(steer_pred, steer_gt.squeeze()).mean() * self._cfg.learn.control_weights[0]
        throttle_loss = self._criterion(throttle_pred,
                                        throttle_gt.squeeze()).mean() * self._cfg.learn.control_weights[1]
        brake_loss = self._criterion(brake_pred, brake_gt.squeeze()).mean() * self._cfg.learn.control_weights[2]

        total_loss = speed_loss + steer_loss + throttle_loss + brake_loss

        return_info = {
            'total_loss': total_loss.item(),
            'speed_loss': speed_loss.item(),
            'steer_loss': steer_loss.item(),
            'throttle_loss': throttle_loss.item(),
            'brake_loss': brake_loss.item(),
            # 'steer_mean': steer_pred.item().mean(),
            # 'throttle_mean': throttle_pred.item().mean(),
            # 'brake_mean': brake_pred.item().mean(),
        }
        return return_info

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    @property
    def validate_mode(self) -> 'CILRSPolicy.eval_function':  # noqa
        return CILRSPolicy.validate_function(
            self._forward_validate,
            self._reset_eval,
            self._get_attribute,
            self._set_attribute,
            self._state_dict_eval,
            self._load_state_dict_eval,
        )
