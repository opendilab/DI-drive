import os
import numpy as np
from pathlib import Path
from typing import Callable, List, Dict

from core.utils.data_utils.data_writter import write_json, write_episode_lmdb
from core.utils.others.image_helper import save_image, is_image
from core.data import BenchmarkDatasetSaver


class CICTBenchmarkDatasetSaver(BenchmarkDatasetSaver):

    def __init__(self, save_dir: str, obs_cfg: Dict, post_process_fn: Callable = None):
        super().__init__(save_dir, obs_cfg, post_process_fn)

    def save_episodes_data(self, episodes_data: List, start_episode: int):
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
            CICTBenchmarkDatasetSaver._make_episode_path(episode_path, episode_data['env_param'])
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
                    observations['velocity'],
                    observations['angular_velocity'],
                    observations['rotation'],
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
                data.append((measurements, sensor_data, {}, others))
                #print(sensor_data.keys(), others.keys())
            CICTBenchmarkDatasetSaver._save_episode_data(episode_path, data)

    @staticmethod
    def _save_episode_data(episode_path, data):
        write_episode_lmdb(episode_path, data)
        for i, x in enumerate(data):
            sensor_data = x[1]
            for k, v in sensor_data.items():
                save_image(os.path.join(episode_path, "%s_%05d.png" % (k, i)), v)

            lidar_data = x[3]['lidar']
            np.save(os.path.join(episode_path, "lidar_%05d.npy" % i), lidar_data)

            waypoint_list = x[3]['waypoint_list']
            np.save(os.path.join(episode_path, "waypoints_%05d.npy" % i), waypoint_list)

            if 'direction_list' in x[3].keys():
                direction_list = x[3]['direction_list']
                np.save(os.path.join(episode_path, "direction_%05d.npy" % i), direction_list)
