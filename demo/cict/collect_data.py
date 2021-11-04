import os
from functools import partial
from tqdm import tqdm
from easydict import EasyDict
from ding.utils.default_helper import deep_merge_dicts

from core.data import CarlaBenchmarkCollector
from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.policy import AutoPIDPolicy
from core.utils.others.tcp_helper import parse_carla_tcp
from ding.envs import SyncSubprocessEnvManager
from post import destination, destination2, save_as_npy, config
from cict_datasaver import CICTBenchmarkDatasetSaver

main_config = EasyDict(config)


def wrapped_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    return CarlaEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg)


def cict_post_process_fn(observations):
    sensor_data = {}
    others = {}
    for key, value in observations.items():
        if key in ['rgb', 'depth', 'segmentation', 'bev']:
            sensor_data[key] = value
        elif key in ['lidar', 'waypoint_list', 'direction_list']:
            others[key] = value
    return sensor_data, others


def post_process(datasets_path):
    epi_folder = [x for x in os.listdir(datasets_path) if x.startswith('epi')]

    for episode_path in tqdm(epi_folder):
        destination(datasets_path, episode_path)
        destination2(datasets_path, episode_path)
        save_as_npy(datasets_path, episode_path)


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

    if not os.path.exists(cfg.policy.collect.dir_path):
        os.makedirs(cfg.policy.collect.dir_path)

    collected_episodes = 0
    saver = CICTBenchmarkDatasetSaver(cfg.policy.collect.dir_path, cfg.env.simulator.obs, post_process_fn=cict_post_process_fn)
    saver.make_dataset_path(cfg.policy.collect)
    while collected_episodes < cfg.policy.collect.n_episode:
        # Sampling data from environments
        print('start collect data')
        new_data = collector.collect(n_episode=env_num)
        print(new_data[0].keys())
        saver.save_episodes_data(new_data, start_episode=collected_episodes)
        collected_episodes += env_num

    collector_env.close()
    post_process(cfg.policy.collect.dir_path)


if __name__ == '__main__':
    main(main_config)
