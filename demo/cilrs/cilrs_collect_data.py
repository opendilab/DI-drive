import os
from functools import partial

import PIL
import lmdb
import numpy as np
from ding.envs import SyncSubprocessEnvManager
from ding.utils.default_helper import deep_merge_dicts
from easydict import EasyDict
from tqdm import tqdm

from core.data import CarlaBenchmarkCollector, BenchmarkDatasetSaver
from core.envs import SimpleCarlaEnv, DriveEnvWrapper
from core.policy import AutoPIDPolicy
from core.utils.others.tcp_helper import parse_carla_tcp

config = dict(
    env=dict(
        env_num=5,
        simulator=dict(
            disable_two_wheels=True,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[400, 300],
                    position=[1.3, 0.0, 2.3],
                    fov=100,
                ),
            ),
            verbose=True,
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        ran_light_is_failure=True,
        manager=dict(
            auto_reset=False,
            shared_memory=False,
            context='spawn',
            max_retry=1,
        ),
        wrapper=dict(
            speed_factor=25.,
            scale=1,
            crop=256,
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9010, 2]),
    ],
    policy=dict(
        target_speed=25,
        tl_threshold=13,
        noise=True,
        noise_kwargs=dict(),
        collect=dict(
            n_episode=100,
            dir_path='./datasets_train/cilrs_datasets_train',
            preloads_name='cilrs_datasets_train.npy',
            collector=dict(
                suite='FullTown01-v1',
                nocrash=True,
            ),
        )
    ),
)

main_config = EasyDict(config)


def cilrs_postprocess(observasion, scale=1, crop=256):
    rgb = observasion['rgb'].copy()
    im = PIL.Image.fromarray(rgb)
    (width, height) = (int(im.width // scale), int(im.height // scale))
    rgb = im.resize((width, height))
    rgb = np.asarray(rgb)
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    rgb = rgb[start_x:start_x + crop, start_y:start_y + crop]
    sensor_data = {'rgb': rgb}
    others = {}
    return sensor_data, others


def wrapped_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    return DriveEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg)


def post_process(config):
    epi_folder = [x for x in os.listdir(config.policy.collect.dir_path) if x.startswith('epi')]

    all_img_list = []
    all_mea_list = []

    for item in tqdm(epi_folder):
        lmdb_file = lmdb.open(os.path.join(config.policy.collect.dir_path, item, 'measurements.lmdb')).begin(write=False)
        png_files = [
            x for x in os.listdir(os.path.join(config.policy.collect.dir_path, item)) if (x.endswith('png') and x.startswith('rgb'))
        ]
        png_files.sort()
        for png_file in png_files:
            index = png_file.split('_')[1].split('.')[0]
            measurements = np.frombuffer(lmdb_file.get(('measurements_%05d' % int(index)).encode()), np.float32)
            data = {}
            data['control'] = np.array([measurements[15], measurements[16], measurements[17]]).astype(np.float32)
            data['speed'] = measurements[10] / config.env.wrapper.speed_factor
            data['command'] = float(measurements[11])
            new_dict = {}
            new_dict['brake'] = data['control'][2]
            new_dict['steer'] = (data['control'][0] + 1) / 2
            new_dict['throttle'] = data['control'][1]
            new_dict['speed'] = data['speed']
            new_dict['command'] = data['command']
            all_img_list.append(os.path.join(item, png_file))
            all_mea_list.append(new_dict)
    if not os.path.exists('_preloads'):
        os.mkdir('_preloads')
    np.save('_preloads/{}'.format(config.policy.collect.preloads_name), [all_img_list, all_mea_list])


def main(cfg, seed=0):
    cfg.env.manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env.manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env.env_num
    assert len(tcp_list) >= env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(env_num, len(tcp_list))

    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env.manager,
    )

    policy = AutoPIDPolicy(cfg.policy)

    collector = CarlaBenchmarkCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode)

    if not os.path.exists(cfg.policy.collect.dir_path):
        os.makedirs(cfg.policy.collect.dir_path)

    collected_episodes = 0
    data_postprocess = lambda x: cilrs_postprocess(x, scale=cfg.env.wrapper.scale, crop=cfg.env.wrapper.crop)
    saver = BenchmarkDatasetSaver(cfg.policy.collect.dir_path, cfg.env.simulator.obs, data_postprocess)
    print('[MAIN] Start collecting data')
    saver.make_dataset_path(cfg.policy.collect)
    while collected_episodes < cfg.policy.collect.n_episode:
        # Sampling data from environments
        n_episode = min(cfg.policy.collect.n_episode - collected_episodes, env_num * 2)
        new_data = collector.collect(n_episode=n_episode)
        saver.save_episodes_data(new_data, start_episode=collected_episodes)
        collected_episodes += n_episode
        print('[MAIN] Current collected: ', collected_episodes, '/', cfg.policy.collect.n_episode)

    collector_env.close()
    saver.make_index()
    print('[MAIN] Making preloads')
    post_process(cfg)


if __name__ == '__main__':
    main(main_config)
