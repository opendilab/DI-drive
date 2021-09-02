import numpy as np
import torch

from ding.torch_utils import to_dtype, to_ndarray


def pack_birdview(data, packbit=False):
    if isinstance(data, dict):
        if 'obs' in data:
            pack_birdview(data['obs'])
        if 'next_obs' in data:
            pack_birdview(data['next_obs'])
        if 'birdview' in data:
            bev = data['birdview']
            if isinstance(bev, np.ndarray):
                bev = to_ndarray(bev, dtype=np.uint8)
            elif isinstance(bev, torch.Tensor):
                bev = to_dtype(bev, dtype=torch.uint8)
            data['birdview'] = bev
        if 'obs' not in data and 'next_obs' not in data and 'birdview' not in data:
            for value in data.values():
                pack_birdview(value)
    if isinstance(data, list):
        for item in data:
            pack_birdview(item)


def unpack_birdview(data, unpackbit=False, shape=[-1]):
    if isinstance(data, dict):
        if 'obs' in data:
            unpack_birdview(data['obs'])
        if 'next_obs' in data:
            unpack_birdview(data['next_obs'])
        if 'birdview' in data:
            bev = data['birdview']
            if isinstance(bev, np.ndarray):
                bev = to_ndarray(bev, dtype=np.float32)
            elif isinstance(bev, torch.Tensor):
                bev = to_dtype(bev, dtype=torch.float32)
            data['birdview'] = bev
        if 'obs' not in data and 'next_obs' not in data and 'birdview' not in data:
            for value in data.values():
                unpack_birdview(value)
    if isinstance(data, list):
        for item in data:
            unpack_birdview(item)
