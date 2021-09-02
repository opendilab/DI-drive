import os
from functools import partial

import PIL
import lmdb
import numpy as np
from ding.envs import SyncSubprocessEnvManager
from ding.utils.default_helper import deep_merge_dicts
from easydict import EasyDict
from tqdm import tqdm

from core.data import CarlaBenchmarkCollector
from core.data.dataset_saver import BenchmarkDatasetSaver
from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.policy import AutoPIDPolicy
from core.utils.others.tcp_helper import parse_carla_tcp

config = dict(
    env=dict(
        simulator=dict(
            disable_two_wheels=True,
            waypoint_num=32,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[800, 600],
                    position=[2.0, 0.0, 1.4],
                    rotation=[-15, 0, 0],
                ),
            ),

        ),
        col_is_failure=True,
        stuck_is_failure=True,
    ),
    env_num=1,
    episode_nums=5,
    env_manager=dict(
        auto_reset=False,
        shared_memory=False,
    ),
    env_wrapper=dict(),
    collector=dict(
        suite='FullTown01-v1',
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[5000, 5002, 2]),
    ],
    policy=dict(
        target_speed=25,
        noise=True,
    ),
    dir_path='./datasets_train/cils_datasets_train',
)

main_config = EasyDict(config)


def cils_postprocess(sensor_data, *args):
    rgb = sensor_data['rgb'].copy()
    rgb = rgb[115:500, :, :]
    im = PIL.Image.fromarray(rgb)
    rgb = np.array(im.resize([200, 88], PIL.Image.BICUBIC))
    sensor_data = {'rgb': rgb}
    others = {}
    return sensor_data, others


def wrapped_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    return CarlaEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg)


def post_process(datasets_path):
    epi_folder = [x for x in os.listdir(datasets_path) if x.startswith('epi')]

    all_img_list = []
    all_mea_list = []

    for item in tqdm(epi_folder):
        lmdb_file = lmdb.open(os.path.join(datasets_path, item, 'measurements.lmdb')).begin(write=False)
        png_file = [
            x for x in os.listdir(os.path.join(datasets_path, item)) if (x.endswith('png') and x.startswith('rgb'))
        ]
        png_file.sort()
        for k in tqdm(png_file):
            index = k.split('_')[1].split('.')[0]
            measurements = np.frombuffer(lmdb_file.get(('measurements_%05d' % int(index)).encode()), np.float32)
            data = {}
            data['control'] = np.array([measurements[12], measurements[13], measurements[14]]).astype(np.float32)
            data['speed'] = measurements[10] / 25.
            data['command'] = float(measurements[11])
            new_dict = {}
            new_dict['brake'] = data['control'][2]
            new_dict['steer'] = data['control'][0]
            new_dict['throttle'] = data['control'][1]
            new_dict['speed_module'] = data['speed']
            new_dict['directions'] = data['command'] + 1.0
            all_img_list.append(os.path.join(item, k))
            all_mea_list.append(new_dict)
    if not os.path.exists('_preloads'):
        os.mkdir('_preloads')
    np.save('_preloads/50hours_cils_datasets_train.npy', [all_img_list, all_mea_list])


def main(cfg, seed=0):
    cfg.env_manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env_manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env_num

    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env_wrapper, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env_manager,
    )
    collector_env.seed(seed)

    policy = AutoPIDPolicy(cfg.policy)

    collector = CarlaBenchmarkCollector(cfg.collector, collector_env, policy.collect_mode)

    if not os.path.exists(cfg.dir_path):
        os.makedirs(cfg.dir_path)

    collected_episodes = -1
    saver = BenchmarkDatasetSaver(cfg.dir_path, cfg.env.simulator.obs, cils_postprocess)
    while collected_episodes < cfg.episode_nums:
        # Sampling data from environments
        print('start collect data')
        new_data = collector.collect(n_episode=env_num)
        print(new_data[0].keys())
        collected_episodes += env_num
        saver.save_episodes_data(new_data, start_episode=collected_episodes)

    collector_env.close()
    post_process(cfg.dir_path)


if __name__ == '__main__':
    main(main_config)
