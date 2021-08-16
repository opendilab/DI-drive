import collections
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.utils.data_utils import splitter


def parse_remove_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'remove'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key

    return name, conf_dict


class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(
        self,
        root_dir,
        cfg,
        transform=None,
    ):
        # Setting the root directory for this dataset
        self.root_dir = root_dir
        self.cfg = cfg

        preload_name = str(cfg.NUMBER_OF_HOURS) + 'hours_' + cfg.TRAIN_DATASET_NAME
        # We add to the preload name all the remove labels
        if cfg.REMOVE is not None and cfg.REMOVE != "None":
            name, self._remove_params = parse_remove_configuration(cfg.REMOVE)
            self.preload_name = preload_name + '_' + name
            self._check_remove_function = getattr(splitter, name)
        else:
            self._check_remove_function = lambda _, __: False
            self._remove_params = []
            self.preload_name = preload_name

        print("preload Name ", self.preload_name)

        if self.preload_name is not None and os.path.exists(os.path.join('_preloads', self.preload_name + '.npy')):
            print(" Loading from NPY ")
            self.sensor_data_names, self.measurements = np.load(
                os.path.join('_preloads', self.preload_name + '.npy'), allow_pickle=True
            )
            print(self.sensor_data_names)
        else:
            raise ValueError('The .npy file doesn\'t exists, please specify the right file path.')

        print("preload Name ", self.preload_name)

        self.transform = transform
        self.batch_read_number = 0

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.

        Args:
            index:

        Returns:

        """
        try:
            img_path = os.path.join(
                self.root_dir, self.sensor_data_names[index].split('/')[-2],
                self.sensor_data_names[index].split('/')[-1]
            )

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # Apply the image transformation
            if self.transform is not None:
                boost = 1
                img = self.transform(self.batch_read_number * boost, img)
            else:
                img = img.transpose(2, 0, 1)

            img = img.astype(np.float)
            img = torch.from_numpy(img).type(torch.FloatTensor)
            img = img / 255.

            measurements = self.measurements[index].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([
                    v,
                ]))
                measurements[k] = v.float()

            measurements['rgb'] = img

            self.batch_read_number += 1
        except AttributeError:
            print("Blank IMAGE")

            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([
                    v,
                ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb'] = torch.from_numpy(np.zeros((3, 88, 200))).type(torch.FloatTensor)

        return measurements

    def extract_targets(self, data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in self.cfg.TARGETS:
            targets_vec.append(data[target_name])

        return torch.cat(targets_vec, 1)

    def extract_inputs(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in self.cfg.INPUTS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)
