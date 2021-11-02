import os
from easydict import EasyDict
from pathlib import Path
from functools import partial

import lmdb
from PIL import Image
import numpy as np

from core.data import CarlaBenchmarkCollector
from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.policy import AutoPIDPolicy
from core.utils.others.tcp_helper import parse_carla_tcp
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.utils.default_helper import deep_merge_dicts

config = dict(
    env=dict(
        env_num=5,
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
                    size=[288, 288],
                    fov=100,
                    position=[1.5, 0.0, 2.4],
                    rotation=[0.0, 0.0, 0.0],
                ),
                dict(
                    name='segmentation',
                    type='segmentation',
                    size=[256, 256],
                    fov=100,
                    position=[1.5, 0.0, 2.4],
                    rotation=[0.0, 0.0, 0.0],
                )
            ),
            aug=dict(
                position_range=[2.0, 0.0, 0.0],
                rotation_range=[0.0, 30.0, 0.0],
            ),
            verbose=True,
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        manager=dict(
            auto_reset=False,
            shared_memory=False,
            context='spawn',
            max_retry=1,
        ),
        wrapper=dict(),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9010, 2]),
    ],
    policy=dict(
        target_speed=25,
        noise=False,
        collect=dict(
            save_dir='dataset/',
            n_episode=50,
            collector=dict()
        ),
    ),
)

main_config = EasyDict(config)


def write_episode_data(episode_path, episode_data):
    lmdb_store_keys = ['aug_rot', 'aug_pos', 'is_junction', 'tl_dis', 'tl_state']
    sensor_keys = ['segmentation', 'rgb']
    lmdb_env = lmdb.open(os.path.join(episode_path, "measurements.lmdb"), map_size=1e10)
    with lmdb_env.begin(write=True) as txn:
        txn.put('len'.encode(), str(len(episode_data)).encode())
        for i, x in enumerate(episode_data):
            data = episode_data[i]['obs']
            data['aug_rot'] = data['aug']['aug_rot']
            data['aug_pos'] = data['aug']['aug_pos']
            for key in lmdb_store_keys:
                txn.put(('%s_%05d' % (key, i)).encode(), np.ascontiguousarray(data[key]).astype(np.float32))
            for key in sensor_keys:
                image = Image.fromarray(data[key])
                image.save(os.path.join(episode_path, "%s_%05d.png" % (key, i)))


def wrapped_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    return CarlaEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg)


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
    collector_env.seed(seed)

    policy = AutoPIDPolicy(cfg.policy)

    collector = CarlaBenchmarkCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode)

    if not os.path.exists(cfg.policy.collect.save_dir):
        os.mkdir(cfg.policy.collect.save_dir)

    collected_episodes = 0

    while collected_episodes < cfg.policy.collect.n_episode:
        # Sampling data from environments
        print('start collect data')
        new_data = collector.collect(n_episode=env_num)
        for i in range(len(new_data)):
            collected_episodes += 1
            episode_path = Path(cfg.policy.collect.save_dir).joinpath('episode_%05d' % collected_episodes)
            if not os.path.exists(episode_path):
                os.mkdir(episode_path)
            write_episode_data(episode_path, new_data[i]['data'])
            if collected_episodes > cfg.policy.collect.n_episode:
                break

    collector_env.close()


if __name__ == '__main__':
    main(main_config)
