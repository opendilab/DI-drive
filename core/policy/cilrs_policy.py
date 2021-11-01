from typing import Dict, List, Optional, Any
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F

from ding.utils.data import default_collate, default_decollate
from ding.torch_utils import to_device
from core.models import CILRSModel
from .base_carla_policy import BaseCarlaPolicy


class CILRSPolicy(BaseCarlaPolicy):
    """
    CILRS driving policy. It has a CILRS NN model which can handle
    observations from several environments by collating data into batch. It contains 2
    modes: `eval` and `learn`. The learn mode will calculate all losses, but will not
    back-propregate it. In `eval` mode, the output control signal will be postprocessed to
    standard control signal in Carla, and it can avoid stopping in the staring ticks.

    :Arguments:
        - cfg (Dict): Config Dict.

    :Interfaces:
        reset, forward
    """

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

    def __init__(
            self,
            cfg: Dict,
    ) -> None:
        super().__init__(cfg, enable_field=[])
        self._enable_field = ['eval', 'learn']
        self._cuda = self._cfg.cuda
        self._max_throttle = self._cfg.max_throttle
        self._model = CILRSModel(**self._cfg.model)
        if self._cuda:
            self._model.cuda()

        for field in self._enable_field:
            getattr(self, '_init_' + field)()

    def _process_sensors(self, sensor: np.ndarray) -> np.ndarray:
        sensor = sensor[:, :, ::-1]  # BGR->RGB
        sensor = np.transpose(sensor, (2, 0, 1))
        sensor = sensor / 255.0

        return sensor

    def _process_model_outputs(self, data: Dict, output: List) -> List:
        action = []
        for i, d in enumerate(data.values()):
            control_pred = output[i][0]
            steer = control_pred[0] * 2 - 1.  # convert from [0,1] to [-1,1]
            throttle = min(control_pred[1], self._max_throttle)
            brake = control_pred[2]
            if d['tick'] < 20 and d['speed'] < 0.1:
                throttle = self._max_throttle
                brake = 0
            if brake < 0.05:
                brake = 0
            action.append({'steer': steer, 'throttle': throttle, 'brake': brake})
        return action

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        """
        Reset policy of `eval` mode. It will change the NN model into 'eval' mode.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        self._model.eval()

    @torch.no_grad()
    def _forward_eval(self, data: Dict) -> Dict[str, Any]:
        """
        Running forward to get control signal of `eval` mode.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: Control and waypoints dict stored in values for each provided env id.
        """
        data_id = list(data.keys())

        new_data = dict()
        for id in data.keys():
            new_data[id] = dict()
            new_data[id]['rgb'] = self._process_sensors(data[id]['rgb'].numpy())
            new_data[id]['command'] = data[id]['command']
            new_data[id]['speed'] = data[id]['speed']

        new_data = default_collate(list(new_data.values()))
        if self._cuda:
            new_data = to_device(new_data, 'cuda')

        embedding = self._model.encode([new_data['rgb']])
        output = self._model(embedding, new_data['speed'], new_data['command'])
        if self._cuda:
            output = to_device(output, 'cpu')

        actions = default_decollate(output)
        actions = self._process_model_outputs(data, actions)
        return {i: {'action': d} for i, d in zip(data_id, actions)}

    def _init_learn(self) -> None:
        if self._cfg.learn.loss == 'l1':
            self._criterion = F.l1_loss
        elif self._cfg.policy.learn.loss == 'l2':
            self._criterion = F.mse_loss

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        """
        Reset policy of `learn` mode. It will change the NN model into 'train' mode.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        self._model.train()

    def _forward_learn(self, data: Dict) -> Dict[str, Any]:
        """
        Running forward of `learn` mode to get loss.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: information about training loss.
        """
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
            'total_loss': total_loss,
            'speed_loss': speed_loss,
            'steer_loss': steer_loss,
            'throttle_loss': throttle_loss,
            'brake_loss': brake_loss,
            # 'steer_mean': steer_pred.item().mean(),
            # 'throttle_mean': throttle_pred.item().mean(),
            # 'brake_mean': brake_pred.item().mean(),
        }

        return return_info
