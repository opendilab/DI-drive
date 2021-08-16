import os
import collections
import math
import copy
import random
import numpy as np

import torch
import cv2
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.special import comb

from core.utils.data_utils import splitter


class Bezier(object):

    def __init__(self, time_list, xy_list, v0, vf=(1e-6, 1e-6)):
        self.t0 = time_list[0]
        self.t_span = time_list[-1] - time_list[0]
        time_array = (np.array(time_list) - self.t0) / self.t_span
        self.p0 = xy_list[0].reshape(2, 1)

        point_array = np.stack(xy_list, axis=1) - self.p0
        n = point_array.shape[1] + 1
        p0 = point_array[:, 0] + np.array(v0) / n
        pf = point_array[:, -1] - np.array(vf) / n
        point_array = np.insert(point_array, 1, values=p0, axis=1)
        point_array = np.insert(point_array, -1, values=pf, axis=1)

        self.point_array = point_array
        self.expand_point_array = Bezier.expand_control_points(point_array)

    @staticmethod
    def expand_control_points(point_array):
        point_array_expand = copy.deepcopy(point_array)
        size = point_array.shape[1]
        assert size >= 3
        for i in range(1, size - 3):
            p0, p1, p2 = point_array[:, i], point_array[:, i + 1], point_array[:, i + 2]
            norm1, norm2 = np.linalg.norm(p0 - p1), np.linalg.norm(p2 - p1)
            pc = p1 - 0.5 * np.sqrt(norm1 * norm2) * ((p0 - p1) / norm1 + (p2 - p1) / norm2)
            point_array_expand[:, i + 1] = pc
        return point_array_expand

    @staticmethod
    def bernstein(t, i, n):
        return comb(n, i) * t ** i * (1 - t) ** (n - i)

    @staticmethod
    def bezier_curve(t, point_array, bias=0):
        t = np.clip(t, 0, 1)
        n = point_array.shape[1] - 1
        p = np.array([0., 0.]).reshape(2, 1)
        size = len(t) if isinstance(t, np.ndarray) else 1
        p = np.zeros((2, size))
        new_point_array = np.diff(point_array, n=bias, axis=1)
        for i in range(n + 1 - bias):
            p += new_point_array[:, i][:, np.newaxis] * Bezier.bernstein(t, i, n - bias) * n ** bias
        return p

    def position(self, time, expand=True):
        t = (time - self.t0) / self.t_span
        t = np.clip(t, 0, 1)
        p = self.expand_point_array if expand else self.point_array
        return Bezier.bezier_curve(t, p) + self.p0

    def velocity(self, time, expand=True):
        t = (time - self.t0) / self.t_span
        t = np.clip(t, 0, 1)
        p = self.expand_point_array if expand else self.point_array
        return Bezier.bezier_curve(t, p, bias=1)

    def acceleration(self, time, expand=True):
        t = (time - self.t0) / self.t_span
        t = np.clip(t, 0, 1)
        p = self.expand_point_array if expand else self.point_array
        return Bezier.bezier_curve(t, p, bias=2)


def parse_remove_configuration(configuration):
    """
    Turns the configuration line of splitting into a name and a set of params.
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


class CictDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, cfg, img_transform=None, dest_transform=None, pm_transform=None):
        # Setting the root directory for this dataset
        self.root_dir = root_dir
        self.cfg = cfg
        self.img_height = cfg.IMG_HEIGHT
        self.img_width = cfg.IMG_WIDTH

        preload_name = ['episode_%05d.npy' % i for i in range(cfg.START_EPISODE, cfg.END_EPISODE)]
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
        self.img_names = dict()
        self.dest_names = dict()
        self.pm_names = dict()
        self.measurements = dict()
        self.sample_nums = 0
        self.infos = list()
        for name in self.preload_name:
            if name is not None and os.path.exists(os.path.join(cfg.PREFIX, name)):
                print(" Loading from NPY ")
                img_name, dest_name, dest_name2, pm_name, _, measurement = np.load(
                    os.path.join(cfg.PREFIX, name), allow_pickle=True
                )
                print("The number of samples in %s: %d" % (name, len(pm_name)))
                episode_name = name.split('.')[0]
                info = [(episode_name, i) for i in range(len(pm_name))]
                self.img_names[episode_name] = img_name
                if cfg.DEST == 0:
                    self.dest_names[episode_name] = dest_name
                else:
                    self.dest_names[episode_name] = dest_name2
                self.pm_names[episode_name] = pm_name
                self.infos.extend(info)
                self.measurements[episode_name] = measurement
                self.sample_nums += len(pm_name)

        self.img_transform = transforms.Compose(img_transform) if img_transform is not None else None
        self.dest_transform = transforms.Compose(dest_transform) if dest_transform is not None else None
        self.pm_transform = transforms.Compose(pm_transform) if pm_transform is not None else None

        self.batch_read_number = 0

    def __len__(self):
        return int(self.sample_nums)

    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.

        Args:
            index:

        Returns:

        """
        measurements = dict()

        try:
            episode_name, ind = self.get_info(index)
            img_path = os.path.join(self.root_dir, self.img_names[episode_name][ind])
            dest_path = os.path.join(self.root_dir, self.dest_names[episode_name][ind])
            pm_path = os.path.join(self.root_dir, self.pm_names[episode_name][ind])
            fake_dest_id = random.sample(list(range(len(self.pm_names[episode_name]))), 1)[0]
            while (fake_dest_id == index):
                fake_dest_id = random.sample(list(range(len(self.pm_names[episode_name]))), 1)[0]
            fake_dest_path = os.path.join(self.root_dir, self.dest_names[episode_name][fake_dest_id])

            img = Image.open(img_path).convert("RGB")
            dest = Image.open(dest_path).convert("RGB")
            fake_dest = Image.open(fake_dest_path).convert("RGB")
            pm = Image.open(pm_path).convert("L")
            #print(img.size)

            # Apply the transformation
            img = self.apply_transform(img, self.img_transform)
            dest = self.apply_transform(dest, self.dest_transform)
            fake_dest = self.apply_transform(fake_dest, self.dest_transform)
            pm = self.apply_transform(pm, self.pm_transform)

            measurements['rgb'] = img
            measurements['dest'] = dest
            measurements['fake_dest'] = fake_dest
            measurements['pm'] = pm
            measurements['command'] = torch.LongTensor([0])
            #measurements['command'] = torch.LongTensor([self.measurements[episode_name][ind]['command']])

            self.batch_read_number += 1
        except AttributeError:
            print("Blank IMAGE")

            measurements['rgb'] = torch.zeros(3, self.cfg.SENSORS['rgb'][1], self.cfg.SENSORS['rgb'][2]).float()
            measurements['dest'] = torch.zeros(3, self.cfg.SENSORS['rgb'][1], self.cfg.SENSORS['rgb'][2]).float()
            measurements['fake_dest'] = torch.zeros(3, self.cfg.SENSORS['rgb'][1], self.cfg.SENSORS['rgb'][2]).float()
            measurements['pm'] = torch.zeros(1, self.cfg.SENSORS['rgb'][1], self.cfg.SENSORS['rgb'][2]).float()
            measurements['command'] = torch.LongTensor([0])

        return measurements

    def get_info(self, index):
        return self.infos[index]

    def apply_transform(self, x, transforms):
        if transforms is not None:
            x = transforms(x)
        else:
            x = cv2.resize(x, (self.cfg.img_height, self.cfg.img_width), interpolation=cv2.INTER_CUBIC)
            x = x.transpose(2, 0, 1)
            x = torch.from_numpy(x).type(torch.FloatTensor)
            x = x / 255.
            x = 2 * x - 1

        return x

    def is_measurement_partof_experiment(self, measurement_data):

        # If the measurement data is not removable is because it is part of this experiment data
        return not self._check_remove_function(measurement_data, self._remove_params)

    def sample_weights(self):
        command_num = np.array([0., 0., 0., 0.])
        for info in self.infos:
            command_num[int(self.measurements[info[0]][info[1]]['option'])] += 1
        weights = []
        command_num = np.sum(command_num) / command_num
        for k, v in self.measurements.items():

            for i in range(len(self.pm_names[k])):
                weights.append(command_num[int(v[i]['option'])])
        return weights


class PathDataset(Dataset):
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

        self.max_dist = cfg.MAX_DIST
        self.max_t = cfg.MAX_T
        self.img_step = cfg.IMG_STEP
        self.img_height = cfg.IMG_HEIGHT
        self.img_width = cfg.IMG_WIDTH
        self.pred_len = cfg.PRED_LEN
        self.start = 20

        preload_name = ['episode_%05d.npy' % i for i in range(cfg.START_EPISODE, cfg.END_EPISODE)]
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
        self.ipm_names = dict()
        self.measurements = dict()
        self.sample_nums = list()
        self.infos = list()
        for name in self.preload_name:
            if name is not None and os.path.exists(os.path.join(cfg.PREFIX, name)):
                print(" Loading from NPY ")
                img_name, _, _, _, ipm_name, measurement = np.load(os.path.join(cfg.PREFIX, name), allow_pickle=True)
                print("The number of samples in %s: %d" % (name, len(ipm_name)))
                max_num = len(img_name)
                episode_name = name.split('.')[0]
                info = [(episode_name, i, max_num) for i in range(self.start, len(ipm_name))]
                self.ipm_names[episode_name] = ipm_name
                self.infos.extend(info)
                self.measurements[episode_name] = measurement
                self.sample_nums.append(max(0, len(ipm_name) - self.start))
                print(self.sample_nums[-1], len(info))

        self.transform = transforms.Compose(transform) if transform is not None else None

        self.batch_read_number = 0

    def __len__(self):
        return np.sum(self.sample_nums)

    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.

        Args:
            index:

        Returns:

        """
        while True:
            name, cur_id, max_num = self.get_info(index)
            ipms = []
            for i in range(-9, 1):
                ipm_path = self.ipm_names[name][max(0, cur_id + self.img_step * i)]
                if self.cfg.EVAL:
                    ipm_path = ipm_path.split('/')
                    ipm_path[-1] = 'pred_' + ipm_path[-1]
                    ipm_path = '/'.join(ipm_path)
                ipm_path = os.path.join(self.root_dir, ipm_path)
                #print(ipm_path)
                ipm = Image.open(ipm_path).convert('L')
                ipm = self.apply_transform(ipm, self.transform)
                ipms.append(ipm)
            ipms = torch.stack(ipms, dim=0)
            cur_xy = self.measurements[name][cur_id]['location'][:2]
            cur_t = self.measurements[name][cur_id]['time']
            yaw = np.deg2rad(self.measurements[name][cur_id]['rotation'])[1]

            time_list = []
            xy_list = []
            v_xy_list = []
            a_xy_list = []
            a_list = []
            collision_flag = False
            collision_xy = None
            collision_id = None
            for i in range(cur_id, max_num):
                t = self.measurements[name][i]['time']
                #if t - cur_t > self.max_t:
                if len(time_list) >= self.pred_len:
                    break

                xy_t = self.measurements[name][i]['location'][:2]
                rel_xy = self.coordinate_transform(xy_t, yaw, cur_xy)
                uv = self.xy2uv(rel_xy)

                if not collision_flag and (uv[0] >= 0) and (uv[0] < self.img_height) and (uv[1] >= 0) and (
                        uv[1] < self.img_width):
                    if ipms[-1][0, uv[0], uv[1]] < -0.3:
                        collision_flag = True
                        collision_xy = rel_xy
                        collision_id = i
                        #print(collision_id, collision_xy)

                zero = np.zeros((2, ))
                if collision_flag:
                    xy_list.append(collision_xy)
                    v_xy_list.append(zero)
                    a_xy_list.append(zero)
                    a_list.append(0.0)
                    time_list.append(t)
                else:
                    xy_list.append(rel_xy)
                    v_xy = self.measurements[name][i]['velocity'][:2]
                    rel_v = self.coordinate_transform(v_xy, yaw, zero)
                    a_xy = self.measurements[name][i]['acceleration'][:2]
                    rel_a = self.coordinate_transform(a_xy, yaw, zero)
                    v_xy_list.append(rel_v)
                    a_xy_list.append(rel_a)

                    theta_a = np.arctan2(rel_a[1], rel_a[0])
                    theta_v = np.arctan2(rel_v[1], rel_v[0])
                    sign = np.sign(np.cos(theta_a - theta_v))
                    a = sign * np.sqrt(np.sum(rel_a ** 2))
                    a_list.append(a)
                    time_list.append(t)

            ############

            if collision_flag:
                a_brake = 10
                brake_id = collision_id - cur_id
                for i in range(collision_id - cur_id):
                    xy = xy_list[brake_id]
                    safe_dist = np.sqrt(np.sum((xy - collision_xy) ** 2))
                    v_xy = v_xy_list[brake_id]
                    # d = v ** 2 / (2 * a)
                    brake_dist = np.sum(v_xy ** 2) / (2 * a_brake)
                    if brake_dist < safe_dist:
                        break
                    else:
                        brake_id -= 1

                bz_time = [t for t in time_list[brake_id:(collision_id - cur_id)]]
                if len(bz_time) > 2:
                    bz_xy = [xy_list[brake_id], collision_xy]
                    bz_vxy = v_xy_list[brake_id]
                    bezier = Bezier(bz_time, bz_xy, bz_vxy)
                    time_array = np.linspace(bezier.t0, bezier.t0 + bezier.t_span, len(bz_time))
                    position = bezier.position(time_array)
                    velocity = bezier.velocity(time_array)
                    acceleration = bezier.acceleration(time_array)
                    #print(collision_id, cur_id, brake_id)
                    #print(position)
                    for i in range(brake_id, collision_id - cur_id):
                        xy_list[i] = position[:, i - brake_id]
                        v_xy_list[i] = velocity[:, i - brake_id]
                        a_xy_list[i] = acceleration[:, i - brake_id]
                        a_list[i] = -np.sqrt(np.sum((acceleration[:, i - brake_id]) ** 2))
                        time_list[i] = time_array[i - brake_id]

            ###########
            if len(time_list) == 0:
                continue
            else:
                #label_id = random.sample(range(len(time_list)), 1)[0]
                break

        #label_t = torch.FloatTensor([time_list[label_id] - cur_t]) / self.max_t
        label_t = torch.FloatTensor(time_list) - cur_t / self.max_t

        cur_v = self.measurements[name][cur_id]['velocity'][:2]
        cur_v = np.sqrt(np.sum(cur_v ** 2))
        cur_v = torch.FloatTensor([cur_v])

        #label_xy = torch.from_numpy(xy_list[label_id]).float() / self.max_dist
        #label_vxy = torch.from_numpy(v_xy_list[label_id]).float()
        #label_axy = torch.from_numpy(a_xy_list[label_id]).float()
        #label_a = torch.FloatTensor([a_list[label_id]])
        label_xy = torch.from_numpy(np.stack(xy_list, axis=0)).float() / self.max_dist
        label_vxy = torch.from_numpy(np.stack(v_xy_list, axis=0)).float()
        label_axy = torch.from_numpy(np.stack(a_xy_list, axis=0)).float()
        label_a = torch.FloatTensor(a_list)
        #print(label_xy.shape, label_vxy.shape)
        return {
            'ipms': ipms,
            'cur_v': cur_v,
            'label_t': label_t,
            'label_xy': label_xy,
            'label_vxy': label_vxy,
            'label_axy': label_axy,
            'label_a': label_a
        }

    def get_info(self, index):
        return self.infos[index]

    def apply_transform(self, x, transforms):
        if transforms is not None:
            x = transforms(x)
        else:
            x = cv2.resize(x, (self.cfg.img_height, self.cfg.img_width), interpolation=cv2.INTER_CUBIC)
            x = x.transpose(2, 0, 1)
            x = torch.from_numpy(x).type(torch.FloatTensor)
            x = x / 255.
            x = 2 * x - 1

        return x

    def coordinate_transform(self, xy_t, yaw, xy_0):
        dxy = xy_t - xy_0
        rot_mat = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        xy = np.dot(rot_mat, dxy)
        return xy

    def xy2uv(self, xy):
        pixel_per_meter = float(self.img_height) / 25.
        u = (self.img_height - xy[0] * pixel_per_meter)
        v = (xy[1] * pixel_per_meter + self.img_width // 2)
        return np.array([u, v], dtype=np.int32)

    def is_measurement_partof_experiment(self, measurement_data):

        # If the measurement data is not removable is because it is part of this experiment data
        return not self._check_remove_function(measurement_data, self._remove_params)

    def sample_weights(self):
        command_num = np.array([0., 0., 0., 0.])
        for info in self.infos:
            command_num[int(self.measurements[info[0]][info[1]]['option'])] += 1
        weights = []
        command_num = np.sum(command_num) / command_num
        for info in self.infos:
            weights.append(command_num[int(self.measurements[info[0]][info[1]]['option'])])
        return weights
