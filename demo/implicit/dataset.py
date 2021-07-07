import torch
import lmdb
import random
import os
import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def prase_segmentation(img):
    '''
    Convert 3-channel rgb to 6-channel senmantic mask
    Six classes: moving obstacles, traffic lights, road markers, road, sidewalk, background
    '''
    res = np.zeros((img.shape[0], img.shape[1], 6), dtype=np.float32)
    value = np.sum(img, 2)
    res[value == 300, 0] = 1
    res[value == 142, 0] = 1
    res[value == 440, 1] = 1
    res[value == 441, 2] = 1
    res[value == 420, 3] = 1
    res[value == 511, 4] = 1
    res[:, :, 5] = 1 - np.sum(res, 2)

    return res


class ImplicitDataset(Dataset):

    def __init__(self, dataset_path, folders, max_frames=None, crop_sky=True):
        '''
        Dataset for implicit affordances
        Parameters:
            dataset_path: the root path of dataset
            folders: sub folder which will be used under the root path
            max_frames: the maximum limit for one episode
            crop_sky: crop the sky area of the input image
        '''
        self.dataset_path = dataset_path
        self.max_frames = max_frames
        self.steps_image = [-10, -1, -1, 0]
        self.crop_sky = crop_sky

        self.image_transform = transforms.ToTensor()
        self._name_map = {}
        self.file_map = {}
        self.idx_map = {}
        self._build_dataset(dataset_path, folders)

    def _build_dataset(self, dataset_path, folders):
        n_episodes = 0
        for folder in folders:
            full_path = os.path.join(folder, 'measurements.lmdb')
            print(full_path)
            txn = lmdb.open(
                full_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False
            ).begin(write=False)

            n = int(txn.get('len'.encode()))
            offset = len(self._name_map)

            for i in range(n):
                if self.max_frames and len(self) >= self.max_frames:
                    break

                self._name_map[offset + i] = folder
                self.file_map[offset + i] = txn
                self.idx_map[offset + i] = i

            n_episodes += 1

            if self.max_frames and len(self) >= self.max_frames:
                break

        print('%s: %d frames, %d episodes.' % (dataset_path, len(self), n_episodes))

    def __len__(self):
        return len(self.file_map)

    def __getitem__(self, idx):
        '''
        Returns:
            rgb_images: stacked rgb image data (288 * 288 * 12)
            seg_images: stacked segmentation data (128 * 128 * 24)
            tl_state: the state of traffic light
            is_junction: whether the vehicle is at the junction
            tl_dis_cls: the distance from the next traffic light (divded into 4 catefories)
            aug_delta_yaws: the camera augmentation of the yaw and poisition
        '''
        lmdb_txn = self.file_map[idx]
        index = self.idx_map[idx]
        full_path = self._name_map[idx]

        is_junction = np.frombuffer(lmdb_txn.get(('is_junction_%05d' % index).encode()), np.float32)
        aug_yaw = np.frombuffer(lmdb_txn.get(('aug_rot_%05d' % index).encode()), np.float32)[1]
        aug_delta = np.frombuffer(lmdb_txn.get(('aug_pos_%05d' % index).encode()), np.float32)[0]
        tl_state = np.frombuffer(lmdb_txn.get(('tl_state_%05d' % index).encode()), np.float32)[0]
        tl_dis = np.frombuffer(lmdb_txn.get(('tl_dis_%05d' % index).encode()), np.float32)[0]

        tl_state = int(tl_state)
        is_junction = int(is_junction)

        if tl_state == 0 or tl_state == 1:
            tl_state = 0
        else:
            tl_state = 1

        is_junction = int(is_junction)
        aug_yaw = aug_yaw / 20.0

        if tl_dis < 8:
            tl_dis_cls = 0
        elif tl_dis < 20:
            tl_dis_cls = 1
        elif tl_dis < 50:
            tl_dis_cls = 2
        else:
            tl_dis_cls = 3

        aug_delta_yaws = []
        rgb_images = []
        seg_images = []

        for step in self.steps_image:
            if index + step > -1:
                rgb_image = Image.open(os.path.join(full_path, "rgb_%05d.png" % (index + step)))
                rgb_image = np.array(rgb_image)
                seg_image = Image.open(os.path.join(full_path, "segmentation_%05d.png" % (index + step))
                                       ).resize((128, 128), Image.ANTIALIAS)
                seg_image = prase_segmentation(np.array(seg_image))
                aug_yaw = np.frombuffer(lmdb_txn.get(('aug_rot_%05d' % (index + step)).encode()), np.float32)[1]
                aug_delta = np.frombuffer(lmdb_txn.get(('aug_pos_%05d' % (index + step)).encode()), np.float32)[0]
                aug_delta_yaws.append(aug_delta)
                aug_delta_yaws.append(aug_yaw / 20.0)
            else:
                rgb_image = np.zeros((288, 288, 3), dtype=np.float32)
                seg_image = np.zeros((128, 128, 6), dtype=np.float32)
                aug_delta_yaws.append(0.0)
                aug_delta_yaws.append(0.0)
            rgb_image = self.image_transform(np.array(rgb_image))
            seg_image = self.image_transform(np.array(seg_image))
            rgb_images.append(rgb_image)
            seg_images.append(seg_image)
        rgb_images = torch.cat(rgb_images, 0)
        seg_images = torch.cat(seg_images, 0)
        if self.crop_sky:
            rgb_images = rgb_images[:, 120:, :]
        aug_delta_yaws = torch.tensor(aug_delta_yaws)
        return rgb_images, seg_images, tl_state, is_junction, tl_dis_cls, aug_delta_yaws


def _dataloader(data, batch_size, num_workers, shuffle=True):
    return DataLoader(
        data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=True, pin_memory=True
    )


def get_dataloader(
    dataset_dir,
    crop_sky=True,
    batch_size=32,
    num_workers=32,
    shuffle=True,
):
    '''
    Prepare the dataloader
    Parameters:
        dataset_dir: the root path of dataset
        crop_sky: return rgb_images, seg_images, tl_state, is_junction, tl_dis_cls, aug_delta_yaws
        batch_size: the batchsize for training
        num_workers: the workers of dataloader
        shuffle: shuffle the dataset
    '''

    folders = glob.glob('%s/ep**' % dataset_dir)

    train_folders = []
    val_folders = []

    for folder in folders:
        if random.random() < 0.2:
            val_folders.append(folder)
        else:
            train_folders.append(folder)

    def make_dataset(is_train):
        _dataset_dir = dataset_dir
        _num_workers = 32 if is_train else 0

        dataset_cls = ImplicitDataset

        if is_train:
            data = dataset_cls(_dataset_dir, folders=train_folders, crop_sky=crop_sky)
            data = _dataloader(data, batch_size, _num_workers, shuffle=True)
        else:
            data = dataset_cls(_dataset_dir, folders=val_folders, crop_sky=crop_sky)
            data = _dataloader(data, batch_size, _num_workers, shuffle=False)

        return data

    train = make_dataset(True)
    val = make_dataset(False)

    return train, val
