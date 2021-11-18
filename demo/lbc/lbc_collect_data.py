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
from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.policy import AutoPIDPolicy
from core.utils.others.tcp_helper import parse_carla_tcp

config = dict(
    env=dict(
        env_num=5,
        simulator=dict(
            disable_two_wheels=True,
            planner=dict(
                type='lbc',
                resolution=1,
                threshold_before=7.5,
                threshold_after=5.,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[384, 160],
                    position=[2.0, 0.0, 1.4],
                    fov=90,
                ),
                dict(
                    name='birdview',
                    type='bev',
                    size=[320, 320],
                    pixels_per_meter=5,
                    pixels_ahead_vehicle=100,
                ),
            ),
            verbose=True,
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        wrapper=dict(),
        manager=dict(
            auto_reset=False,
            shared_memory=False,
            context='spawn',
            max_retry=1,
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[5000, 5010, 2]),
    ],
    policy=dict(
        target_speed=25,
        lateral_dict={'K_P': 0.75, 'K_D': 0., 'K_I': 0.05, 'dt': 0.1},
        longitudinal_dict={'K_P': 0.5, 'K_D': 0.1, 'K_I': 0.025},
        noise=True,
        noise_kwargs=dict(),
        collect=dict(
            dir_path='./datasets_train/lbc_datasets_train',
            n_episode=100,
            collector=dict(
                suite='FullTown01-v1',
                nocrash=True,
            ),
        ),
    ),
)

main_config = EasyDict(config)


def lbc_postprocess(observations, *args):
    sensor_data = {}
    sensor_data['rgb'] = observations['rgb']
    sensor_data['birdview'] = observations['birdview'][..., :7]
    others = {}
    return sensor_data, others


def wrapped_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    return CarlaEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg)


def main(cfg, seed=0):
    cfg.env.manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env.manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env.env_num

    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env.manager,
    )
    collector_env.seed(seed)

    policy = AutoPIDPolicy(cfg.policy)

    collector = CarlaBenchmarkCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode)

    if not os.path.exists(cfg.policy.collect.dir_path):
        os.makedirs(cfg.policy.collect.dir_path)

    collected_episodes = 0
    saver = BenchmarkDatasetSaver(cfg.policy.collect.dir_path, cfg.env.simulator.obs, lbc_postprocess)
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


if __name__ == '__main__':
    main(main_config)
