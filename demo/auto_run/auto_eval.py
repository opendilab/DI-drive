'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''

from functools import partial

from easydict import EasyDict
from ding.envs import SyncSubprocessEnvManager
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.eval import CarlaBenchmarkEvaluator
from core.policy import AutoPIDPolicy
from core.utils.others.tcp_helper import parse_carla_tcp

autoeval_config = dict(
    env_num=4,
    env=dict(simulator=dict(
        verbose=False,
        obs=(),
        planner=dict(type='behavior', ),
    ), ),
    env_manager=dict(
        shared_memory=False,
        auto_reset=False,
    ),
    server=[dict(carla_host='localhost', carla_ports=[9000, 9008, 2])],
    eval=dict(
        suite='FullTown01-v0',
        episodes_per_suite=10,
    ),
    policy=dict(target_speed=40, ),
)

main_config = EasyDict(autoeval_config)


def wrapped_env(env_cfg, host, port, tm_port=None):
    return CarlaEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port))


def main(cfg, seed=0):
    cfg.env_manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env_manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env_num

    evaluate_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env_manager,
    )
    evaluate_env.seed(seed)
    set_pkg_seed(seed)
    auto_policy = AutoPIDPolicy(cfg.policy)
    evaluator = CarlaBenchmarkEvaluator(cfg.eval, evaluate_env, auto_policy.eval_mode)
    evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
