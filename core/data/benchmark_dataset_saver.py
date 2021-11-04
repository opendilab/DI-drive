import os
import lmdb
import glob
import numpy as np
from pathlib import Path
from typing import Callable, List, Dict, Optional

from core.utils.data_utils.data_writter import write_json, write_episode_lmdb
from core.utils.others.image_helper import save_image, is_image


def default_post_process_fn(observations):
    sensor_data = {}
    others = {}
    for key, value in observations.items():
        if is_image(value):
            sensor_data[key] = value
    return sensor_data, others


class BenchmarkDatasetSaver():
    """
    Benchmark dataset saver in DI-drive. It can save dataset in standard benchmark dataset form
    defined in DI-drive. User can pass a post-process function to specialize 'sensor_data' and
    'others' saved in dataset.

    :Arguments:
        - save_dir (str): Dataset folder path.
        - obs_cfg (Dict): Observation config dict in simulator.
        - post_process_fn (Callable, optional): Post-process function defined by user. Defaults to None.
        - lmdb_obs (List, optional): Observation types that saved as lmdb rather than image, default to ['lidar', 'bev']

    :Interfaces: make_dataset_path, save_episodes_data, make_index
    """

    def __init__(
            self,
            save_dir: str,
            obs_cfg: Dict,
            post_process_fn: Optional[Callable] = None,
            lmdb_obs: Optional[List] = ['lidar', 'birdview'],
    ) -> None:
        self._save_dir = save_dir
        self._obs_cfg = obs_cfg
        self._post_process_fn = post_process_fn
        self._lmdb_obs_type = lmdb_obs
        if self._post_process_fn is None:
            self._post_process_fn = default_post_process_fn

    def save_episodes_data(self, episodes_data: List, start_episode: int = 0) -> None:
        """
        Save data from several episodes sampled from collector, with 'env_param' and 'data' key
        saved in each episode.

        :Arguments:
            - episode_count (int): Start count of episode to save.
            - episodes_data (List): Saved data of episodes.
        """
        for episode, episode_data in enumerate(episodes_data):
            data = list()
            episode_path = Path(self._save_dir).joinpath('episode_%05d' % (start_episode + episode))
            BenchmarkDatasetSaver._make_episode_path(episode_path, episode_data['env_param'])
            for idx, frame_data in enumerate(episode_data['data']):
                observations = frame_data['obs']
                actions = frame_data['action']
                if 'real_steer' not in actions:
                    actions['real_steer'] = actions['steer']
                    actions['real_throttle'] = actions['throttle']
                    actions['real_brake'] = actions['brake']

                measurements = [
                    observations['tick'],
                    observations['timestamp'],
                    observations['forward_vector'],
                    observations['acceleration'],
                    observations['location'],
                    observations['speed'],
                    observations['command'],
                    actions['steer'],
                    actions['throttle'],
                    actions['brake'],
                    actions['real_steer'],
                    actions['real_throttle'],
                    actions['real_brake'],
                    observations['tl_state'],
                    observations['tl_dis'],
                ]

                measurements = [x if x.shape != () else np.float32([x]) for x in measurements]
                measurements = np.concatenate(measurements, 0)
                sensor_data, others = self._post_process_fn(observations)
                data.append((measurements, sensor_data, others))
            BenchmarkDatasetSaver._save_episode_data(episode_path, data, self._lmdb_obs_type)

    def make_dataset_path(self, dataset_metainfo: Dict = dict()) -> None:
        """
        Make dataset folder and write dataset meta infomation into a json file.

        :Arguments:
            - dataset_metainfo (Dict): the metainfo of datasets
        """
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        obs_types = ['rgb', 'depth', 'segmentation', 'lidar', 'bev']
        obs_metainfo = {}
        for obs_item in self._obs_cfg:
            if obs_item.type in obs_types:
                obs_name = obs_item.name
                obs_item = obs_item.copy()
                obs_item.pop('name')
                obs_metainfo.update({obs_name: obs_item})

        dataset_metainfo.update({'obs': obs_metainfo})

        write_json(os.path.join(self._save_dir, 'metainfo.json'), dataset_metainfo)

    @staticmethod
    def _make_episode_path(episode_path, env_params) -> None:
        os.makedirs(episode_path, exist_ok=True)
        write_json(os.path.join(episode_path, 'episode_metainfo.json'), env_params)

    @staticmethod
    def _save_episode_data(episode_path, data, lmdb_obs_type=None) -> None:
        write_episode_lmdb(episode_path, data, lmdb_obs_type)
        for i, x in enumerate(data):
            sensor_data = x[1]
            for k, v in sensor_data.items():
                if lmdb_obs_type is not None and k in lmdb_obs_type:
                    continue
                else:
                    save_image(os.path.join(episode_path, "%s_%05d.png" % (k, i)), v)

    def make_index(self, command_index: int = 11) -> None:
        """
        Make an index txt file to save all the command of each frame in dataset.

        :Arguments:
            - command_index (int, optional): The index of command in 'measurements.lmdb'. Defaults to 11.
        """
        index_path = os.path.join(self._save_dir, 'index.txt')
        episode_list = glob.glob('%s/episode*' % self._save_dir)
        episode_list = sorted(episode_list)

        with open(index_path, 'w') as index_f:
            for episode_path in episode_list:
                eph = os.path.split(episode_path)[-1]
                txn = lmdb.open(os.path.join(episode_path, 'measurements.lmdb')).begin(write=False)
                n = int(txn.get('len'.encode()))
                for i in range(n):
                    info = ''
                    info += eph + ','
                    measurements = np.frombuffer(txn.get(('measurements_%05d' % i).encode()), np.float32)
                    info += str(i) + ','
                    info += str(int(measurements[command_index])) + '\n'
                    index_f.write(info)
