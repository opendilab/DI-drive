import os
from functools import partial

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
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='birdview',
                    type='bev',
                    size=[320, 320],
                    pixels_per_meter=5,
                    pixels_ahead_vehicle=100,
                ),
            ),
            verbose=False,
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
        dict(carla_host='local_host', carla_ports=[9000, 9010, 2]),
    ],
    policy=dict(
        target_speed=25,
        noise=False,
        collect=dict(
            dir_path='bev_train',
            n_episode=50,
            collector=dict(
                suite=['NoCrashTown01-v3', 'NoCrashTown01-v5'],
                nocrash=True,
                weathers=[1],
            ),
        ),
    ),
)

main_config = EasyDict(config)


def latent_postprocess(observations, *args):
    sensor_data = {}
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
    saver = BenchmarkDatasetSaver(cfg.policy.collect.dir_path, cfg.env.simulator.obs, latent_postprocess)
    saver.make_dataset_path(cfg.policy.collect)
    while collected_episodes < cfg.policy.collect.n_episode:
        # Sampling data from environments
        n_episode = min(cfg.policy.collect.n_episode - collected_episodes, env_num * 2)
        new_data = collector.collect(n_episode=n_episode)
        saver.save_episodes_data(new_data, start_episode=collected_episodes)
        del new_data
        collected_episodes += n_episode
        print('[MAIN] Current collected: ', collected_episodes, '/', cfg.policy.collect.n_episode)

    collector_env.close()


if __name__ == '__main__':
    main(main_config)
