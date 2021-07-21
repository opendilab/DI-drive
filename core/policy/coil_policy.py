from typing import Dict

import numpy as np
import scipy
import scipy.misc
import torch
from ding.utils.data import default_collate, default_decollate
from ding.torch_utils import to_device

from core.models import COILModel
from .base_carla_policy import BaseCarlaPolicy


class CILPolicy(BaseCarlaPolicy):
    config = dict(debug=False, cuda=True)

    def __init__(
            self,
            cfg: Dict,
    ) -> None:
        super().__init__(cfg)
        self._enable_field = set(['collect', 'eval'])
        self._cuda = True
        for field in self._enable_field:
            getattr(self, '_init_' + field)()

    def _process_sensors(self, sensor):
        sensor = sensor[:, :, ::-1]  # BGR->RGB
        sensor = sensor[self._cfg.IMAGE_CUT[0]:self._cfg.IMAGE_CUT[1], :, :]  # crop
        sensor = scipy.misc.imresize(sensor, (self._cfg.SENSORS['rgb'][1], self._cfg.SENSORS['rgb'][2]))

        sensor = np.swapaxes(sensor, 0, 1)
        sensor = np.transpose(sensor, (2, 1, 0))
        sensor = sensor / 255.0

        return sensor

    def _process_model_outputs(self, outputs):

        actions = []
        for output in outputs:

            steer, throttle, brake = output[0], output[1], output[2]
            if brake < 0.05:
                brake = 0.0

            if throttle > brake:
                brake = 0.0

            action = {'steer': float(steer), 'throttle': float(throttle), 'brake': float(brake)}

        actions.append({'action': action})

        return actions

    def _init_eval(self) -> None:
        self._eval_model = COILModel(self._cfg)

    def _forward_eval(self, data: Dict) -> dict:
        data_id = list(data.keys())

        for id in data.keys():
            data[id]['rgb'] = self._process_sensors(data[id]['rgb'].numpy())

        data = default_collate(list(data.values()))

        with torch.no_grad():
            output = self._eval_model.run_step(data)
        if self._cuda:
            output = to_device(output, 'cpu')

        output = default_decollate(output)
        output = self._process_model_outputs(output)
        return {i: d for i, d in zip(data_id, output)}
