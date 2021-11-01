import os
import lmdb
import cv2
import numpy as np
from typing import Any, Dict
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from core.utils.others.image_helper import read_image, draw_msra_gaussian
from core.utils.data_utils import augmenter

PIXEL_OFFSET = 10
PIXELS_PER_METER = 5


def world_to_pixel(x, y, ox, oy, ori_ox, ori_oy, pixels_per_meter=5, offset=(-80, 160), size=320, angle_jitter=15):
    pixel_dx, pixel_dy = (x - ox) * pixels_per_meter, (y - oy) * pixels_per_meter

    pixel_x = pixel_dx * ori_ox + pixel_dy * ori_oy
    pixel_y = -pixel_dx * ori_oy + pixel_dy * ori_ox

    pixel_x = 320 - pixel_x

    return np.array([pixel_x, pixel_y]) + offset


class LBCBirdViewDataset(Dataset):

    def __init__(
            self,
            root_dir,
            img_size=320,
            crop_size=192,
            gap=5,
            n_step=5,
            crop_x_jitter=5,
            crop_y_jitter=5,
            angle_jitter=5,
            down_ratio=4,
            gaussian_radius=1.0,
            max_frames=None
    ) -> None:
        self._root_dir = root_dir
        self._img_size = img_size
        self._crop_size = crop_size
        self._gap = gap
        self._n_step = n_step
        self._crop_x_jitter = crop_x_jitter
        self._crop_y_jitter = crop_y_jitter
        self._angle_jitter = angle_jitter
        self._gaussian_radius = gaussian_radius
        self._down_ratio = down_ratio
        self._max_frames = max_frames
        self.bird_view_transform = transforms.ToTensor()

        epi_folder = [x for x in os.listdir(root_dir) if x.startswith('epi')]

        self._img_list = []
        self._lmdb_list = []
        self._idx_list = []

        for item in epi_folder:
            lmdb_file = lmdb.open(os.path.join(root_dir, item, 'measurements.lmdb')).begin(write=False)
            max_len = int(lmdb_file.get('len'.encode())) - self._gap * self._n_step
            png_files = [
                x for x in os.listdir(os.path.join(root_dir, item)) if (x.endswith('png') and x.startswith('rgb'))
            ]
            png_files.sort()
            for i in range(max_len):
                png_file = png_files[i]
                index = int(png_file.split('_')[1].split('.')[0])
                self._img_list.append(os.path.join(root_dir, item, png_file))
                self._idx_list.append(index)
                self._lmdb_list.append(lmdb_file)

    def __len__(self):
        return len(self._img_list)

    def __getitem__(self, index) -> Any:
        lmdb_txn = self._lmdb_list[index]
        episode_index = self._idx_list[index]

        birdview = np.frombuffer(lmdb_txn.get(('birdview_%05d' % episode_index).encode()),
                                 np.uint8).reshape(320, 320, 7)
        measurement = np.frombuffer(lmdb_txn.get(('measurements_%05d' % episode_index).encode()), np.float32)
        #print(png_file)
        #rgb_image = read_image(png_file).reshape(160, 384, 3)
        vector = measurement[2:4]
        location = measurement[7:10]
        speed = measurement[10] / 3.6
        command = measurement[11]
        ox, oy, oz = location
        ori_ox, ori_oy = vector

        oangle = np.arctan2(ori_oy, ori_ox)
        delta_angle = np.random.randint(-self._angle_jitter, self._angle_jitter + 1)
        dx = np.random.randint(-self._crop_x_jitter, self._crop_x_jitter + 1)
        dy = np.random.randint(0, self._crop_y_jitter + 1) - PIXEL_OFFSET

        o_camx = ox + ori_ox * 2
        o_camy = oy + ori_oy * 2

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

        angle = np.arctan2(ori_oy, ori_ox) + np.deg2rad(delta_angle)
        ori_ox, ori_oy = np.cos(angle), np.sin(angle)

        locations = []
        orientations = []

        for dt in range(self._gap, self._gap * (self._n_step + 1), self._gap):
            f_index = episode_index + dt

            f_measurement = np.frombuffer(lmdb_txn.get(('measurements_%05d' % f_index).encode()), np.float32)
            x, y, z = f_measurement[7:10]
            ori_x, ori_y = f_measurement[2:4]

            pixel_y, pixel_x = world_to_pixel(x, y, ox, oy, ori_ox, ori_oy, size=self._img_size)
            pixel_x = pixel_x - (self._img_size - self._crop_size) // 2
            pixel_y = self._crop_size - (self._img_size - pixel_y) + 70

            pixel_x -= dx
            pixel_y -= dy

            angle = np.arctan2(ori_y, ori_x) - np.arctan2(ori_oy, ori_ox)
            ori_dx, ori_dy = np.cos(angle), np.sin(angle)

            locations.append([pixel_x, pixel_y])
            orientations.append([ori_dx, ori_dy])

        #birdview = self.bird_view_transform(birdview)

        # Create mask
        output_size = self._crop_size // self._down_ratio
        heatmap_mask = np.zeros((self._n_step, output_size, output_size), dtype=np.float32)
        regression_offset = np.zeros((self._n_step, 2), np.float32)
        indices = np.zeros((self._n_step), dtype=np.int64)

        for i, (pixel_x, pixel_y) in enumerate(locations):
            center = np.array([pixel_x / self._down_ratio, pixel_y / self._down_ratio], dtype=np.float32)
            center = np.clip(center, 0, output_size - 1)
            center_int = np.rint(center)

            draw_msra_gaussian(heatmap_mask[i], center_int, self._gaussian_radius)
            regression_offset[i] = center - center_int
            indices[i] = center_int[1] * output_size + center_int[0]

        return {'birdview': birdview, 'location': np.array(locations), 'command': command, 'speed': speed}


class LBCImageDataset(Dataset):

    def __init__(
        self,
        root_dir,
        rgb_shape=(160, 384, 3),
        img_size=320,
        crop_size=192,
        gap=5,
        n_step=5,
        gaussian_radius=1.,
        down_ratio=4,
        # rgb_mean=[0.29813555, 0.31239682, 0.33620676],
        # rgb_std=[0.0668446, 0.06680295, 0.07329721],
        augment_strategy=None,
        batch_read_number=819200,
        batch_aug=1,
    ) -> None:
        self._root_dir = root_dir
        self._img_size = img_size
        self._crop_size = crop_size
        self._gap = gap
        self._n_step = n_step
        self._gaussian_radius = gaussian_radius
        self._down_ratio = down_ratio
        self._batch_read_number = batch_read_number
        self._batch_aug = batch_aug

        print("augment with ", augment_strategy)
        if augment_strategy is not None and augment_strategy != 'None':
            self.augmenter = getattr(augmenter, augment_strategy)
        else:
            self.augmenter = None

        self.rgb_transform = transforms.ToTensor()
        self.bird_view_transform = transforms.ToTensor()

        epi_folder = [x for x in os.listdir(root_dir) if x.startswith('epi')]

        self._img_list = []
        self._lmdb_list = []
        self._idx_list = []

        for item in epi_folder:
            lmdb_file = lmdb.open(os.path.join(root_dir, item, 'measurements.lmdb')).begin(write=False)
            max_len = int(lmdb_file.get('len'.encode())) - self._gap * self._n_step
            png_files = [
                x for x in os.listdir(os.path.join(root_dir, item)) if (x.endswith('png') and x.startswith('rgb'))
            ]
            png_files.sort()
            for i in range(max_len):
                png_file = png_files[i]
                index = int(png_file.split('_')[1].split('.')[0])
                self._img_list.append(os.path.join(root_dir, item, png_file))
                self._idx_list.append(index)
                self._lmdb_list.append(lmdb_file)

    def __len__(self):
        return len(self._img_list)

    def __getitem__(self, index) -> Any:
        lmdb_txn = self._lmdb_list[index]
        episode_index = self._idx_list[index]
        png_file = self._img_list[index]

        birdview = np.frombuffer(lmdb_txn.get(('birdview_%05d' % episode_index).encode()),
                                 np.uint8).reshape(320, 320, 7)
        measurement = np.frombuffer(lmdb_txn.get(('measurements_%05d' % episode_index).encode()), np.float32)
        #print(png_file)
        rgb_image = read_image(png_file).reshape(160, 384, 3)

        if self.augmenter:
            rgb_images = [
                self.augmenter(self._batch_read_number).augment_image(rgb_image) for i in range(self._batch_aug)
            ]
        else:
            rgb_images = [rgb_image for i in range(self._batch_aug)]

        if self._batch_aug == 1:
            rgb_images = rgb_images[0]

        vector = measurement[2:4]
        location = measurement[7:10]
        speed = measurement[10] / 3.6
        command = measurement[11]
        ox, oy, oz = location
        ori_ox, ori_oy = vector

        oangle = np.arctan2(ori_oy, ori_ox)
        delta_angle = 0
        dx = 0
        dy = -PIXEL_OFFSET

        pixel_ox = 160
        pixel_oy = 260

        rot_mat = cv2.getRotationMatrix2D((pixel_ox, pixel_oy), delta_angle, 1.0)
        birdview = cv2.warpAffine(birdview, rot_mat, birdview.shape[1::-1], flags=cv2.INTER_LINEAR)

        # random cropping
        center_x, center_y = 160, 260 - self._crop_size // 2

        birdview = birdview[dy + center_y - self._crop_size // 2:dy + center_y + self._crop_size // 2,
                            dx + center_x - self._crop_size // 2:dx + center_x + self._crop_size // 2]

        angle = np.arctan2(ori_oy, ori_ox) + np.deg2rad(delta_angle)
        ori_ox, ori_oy = np.cos(angle), np.sin(angle)

        locations = []

        for dt in range(self._gap, self._gap * (self._n_step + 1), self._gap):

            f_index = episode_index + dt

            f_measurement = np.frombuffer(lmdb_txn.get(('measurements_%05d' % f_index).encode()), np.float32)
            x, y, z = f_measurement[7:10]
            ori_x, ori_y = f_measurement[2:4]

            pixel_y, pixel_x = world_to_pixel(x, y, ox, oy, ori_ox, ori_oy, size=self._img_size)
            pixel_x = pixel_x - (self._img_size - self._crop_size) // 2
            pixel_y = self._crop_size - (self._img_size - pixel_y) + 70

            pixel_x -= dx
            pixel_y -= dy

            # Coordinate transform

            locations.append([pixel_x, pixel_y])

        if self._batch_aug == 1:
            rgb_images = self.rgb_transform(rgb_images)
        else:
            # if len()
            #     import pdb; pdb.set_trace()
            rgb_images = torch.stack([self.rgb_transform(img) for img in rgb_images])
        birdview = self.bird_view_transform(birdview)

        return {
            'rgb': rgb_images,
            'birdview': birdview,
            'location': np.array(locations),
            'command': command,
            'speed': speed
        }
