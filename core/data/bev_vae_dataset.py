import os
import lmdb
import cv2
import numpy as np
from typing import Any, Dict
import torch
from torchvision import transforms
from torch.utils.data import Dataset

PIXEL_OFFSET = 10


class BeVVAEDataset(Dataset):

    def __init__(
            self,
            root_dir,
            img_size=320,
            crop_size=192,
            crop_x_jitter=5,
            crop_y_jitter=5,
            angle_jitter=5,
            down_ratio=4,
            max_frames=None
    ) -> None:
        self._root_dir = root_dir
        self._img_size = img_size
        self._crop_size = crop_size
        self._crop_x_jitter = crop_x_jitter
        self._crop_y_jitter = crop_y_jitter
        self._angle_jitter = angle_jitter
        self._down_ratio = down_ratio
        self._max_frames = max_frames
        self.bird_view_transform = transforms.ToTensor()

        epi_folder = [x for x in os.listdir(root_dir) if x.startswith('epi')]

        self._lmdb_list = []
        self._idx_list = []

        for item in epi_folder:
            lmdb_file = lmdb.open(os.path.join(root_dir, item, 'measurements.lmdb')).begin(write=False)
            max_len = int(lmdb_file.get('len'.encode()))
            for i in range(max_len):
                self._lmdb_list.append(lmdb_file)
                self._idx_list.append(i)

    def __len__(self):
        return len(self._lmdb_list)

    def __getitem__(self, index) -> Any:
        lmdb_txn = self._lmdb_list[index]
        episode_index = self._idx_list[index]

        birdview = np.frombuffer(lmdb_txn.get(('birdview_%05d' % episode_index).encode()),
                                 np.uint8).reshape(320, 320, 7) * 255

        delta_angle = np.random.randint(-self._angle_jitter, self._angle_jitter + 1)
        dx = np.random.randint(-self._crop_x_jitter, self._crop_x_jitter + 1)
        dy = np.random.randint(0, self._crop_y_jitter + 1) - PIXEL_OFFSET

        pixel_ox = 160
        pixel_oy = 260

        birdview = cv2.warpAffine(
            birdview,
            cv2.getRotationMatrix2D((pixel_ox, pixel_oy), delta_angle, 1.0),
            birdview.shape[1::-1],
            flags=cv2.INTER_LINEAR
        )

        # random cropping
        center_x, center_y = 160, 260 - self._crop_size // 2
        birdview = birdview[dy + center_y - self._crop_size // 2:dy + center_y + self._crop_size // 2,
                            dx + center_x - self._crop_size // 2:dx + center_x + self._crop_size // 2]

        birdview = self.bird_view_transform(birdview)

        return {'birdview': birdview}
