import numpy as np
from typing import List, Dict, Optional
import copy
import scipy.misc
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from ding.utils.data import default_collate, default_decollate
from ding.torch_utils import to_device

from core.models import VehicleCapacController
from core.policy.base_carla_policy import BaseCarlaPolicy
from demo.cict_demo.cict_model import CICTModel
from demo.cict_demo.post import get_map, get_nav, draw_destination, CollectPerspectiveImage, params, Sensor,\
    find_dest_with_fix_length


class CICTPolicy(BaseCarlaPolicy):
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

        sensor = Sensor(params.sensor_config['rgb'])
        self.collect_perspective = CollectPerspectiveImage(params, sensor)
        longitudinal_args = dict(K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03)
        lateral_args = dict(L=2.405, k_k=1.235, k_theta=0.456, k_e=0.11, alpha=1.8)
        self.CapacController = VehicleCapacController(lateral_args, longitudinal_args)

    def _process_data(self, data):

        img_transforms = transforms.Compose(
            [
                transforms.Resize((self._cfg.IMG_HEIGHT, self._cfg.IMG_WIDTH), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dest_transforms = transforms.Compose(
            [
                transforms.Resize((self._cfg.IMG_HEIGHT, self._cfg.IMG_WIDTH), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        #print(data['rgb'].shape, data['rgb'].numpy().shape)
        img = Image.fromarray(np.uint8(data['rgb'].numpy()[:, :, ::-1]))
        img = img_transforms(img)

        location = data['location'].numpy()
        rotation = data['rotation'].numpy()
        waypoint_list = data['waypoint_list'].numpy()
        if self._cfg.DEST == 0:
            origin_map = get_map()
            plan_map = draw_destination(location, waypoint_list, copy.deepcopy(origin_map))
            dest = get_nav(location, rotation, plan_map, town=1)
        else:
            start = np.linalg.norm(waypoint_list[0][:2] - location[:2])
            dest_loc, _ = find_dest_with_fix_length(start, waypoint_list)
            zero = np.zeros((3, 1))
            zero[:2, 0] = dest_loc
            dest = self.collect_perspective.drawDestInImage(zero, location, rotation)
            #print(location, rotation, dest_loc)

        dest = Image.fromarray(dest)
        dest = dest_transforms(dest)

        lidar = data['lidar'].numpy()
        timestamp = data['timestamp'].numpy()

        v = data['velocity']
        theta = np.deg2rad(rotation)[1]
        v = torch.norm(v[:2])

        return {
            'rgb': img.unsqueeze(0),
            'dest': dest.unsqueeze(0),
            'lidar': lidar,
            'cur_v': v.unsqueeze(0),
            'theta': theta,
            'time': timestamp
        }

    def _process_model_outputs(self, outputs):

        action = self.CapacController.forward(outputs[0], outputs[1])

        return action

    def _init_eval(self) -> None:
        self._eval_model = CICTModel(self._cfg)

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._eval_model.clean_buffer()
        self.CapacController.reset()

    def _forward_eval(self, data: Dict) -> dict:
        #print(data.keys())
        obs = self._process_data(data[list(data.keys())[0]])

        output = self._eval_model.run_step(obs)

        action = self._process_model_outputs(output)
        print(action)
        return {list(data.keys())[0]: {'action': action}}
